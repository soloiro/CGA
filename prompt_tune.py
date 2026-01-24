import argparse
import os, sys
import os.path as osp
import random

import torchvision
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision import transforms
from typing import List, Tuple
from sklearn.manifold import TSNE

import clip_models
import utils

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end',
                 learned_cls=False,sim_init = "looks like a",fine_gained_sim = False,gained_sim = False,sim_matrix = None):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.sim_init = sim_init
        self.fine_gained_sim =fine_gained_sim
        self.gained_sim = gained_sim
        self.sim_matrix = sim_matrix


        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            # n_ctx = len(ctx_init.split(" "))
            n_ctx = len(_tokenizer.encode(ctx_init))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)




        if fine_gained_sim or gained_sim:
            sim_init= sim_init.replace("_", " ")
            # n_sim = len(sim_init.split(" "))
            n_sim = len(_tokenizer.encode(sim_init))

            sim_prompt = tokenize(sim_init).to(self.device)
            with torch.no_grad():
                embedding_sim = clip_model.token_embedding(sim_prompt).type(dtype)
            sim_vectors = embedding_sim[0, 1: 1+n_sim, :]
            prompt_sim = sim_vectors
            sim_gen_model = utils.SimMLP(prompt_sim.shape[1],512,prompt_sim.shape[1])
            sim_gen_model = sim_gen_model.half()
        else:
            n_sim = None
            sim_gen_model = None

        self.relationship_model = sim_gen_model
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
            sim_vectors = sim_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        if fine_gained_sim or gained_sim:
            self.sim_init_state = sim_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        if fine_gained_sim or gained_sim:
            self.sim = nn.Parameter(sim_vectors)

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.clip_model = clip_model
        self.n_sim = n_sim

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_prompt(self, args, ent_interval, adv_dic, pesu_label):
        batch_ent = torch.sum(-pesu_label * torch.log(pesu_label + args.epsilon), dim=1).mean(0)
        batch_ent = batch_ent.numpy().tolist()
        # for i in range(len(adv_dic)):
        if (batch_ent >= ent_interval[0] and batch_ent < ent_interval[1]):
            newctx_con = adv_dic[0]
            print(newctx_con)
        elif (batch_ent >= ent_interval[1] and batch_ent < ent_interval[2]):
            newctx_con = adv_dic[1]
            print(newctx_con)
        elif (batch_ent >= ent_interval[2] and batch_ent < ent_interval[3]):
            newctx_con = adv_dic[2]
            print(newctx_con)
        elif (batch_ent >= ent_interval[3] and batch_ent < ent_interval[4]):
            newctx_con = adv_dic[3]
            print(newctx_con)
        elif (batch_ent >= ent_interval[4]):
            newctx_con = adv_dic[3]
            print(newctx_con)
        elif (batch_ent <= ent_interval[0]):
            newctx_con = adv_dic[0]
            print(newctx_con)
        # List_rd = classname
        # space = " "
        # indices = indices.squeeze(0).cpu().numpy().tolist()
        # newctx = [List_rd[i] for i in indices]
        # newctx_con = space.join(newctx)
        n_ctx = len(newctx_con.split(" "))
        prompt = tokenize(newctx_con).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        # self.ctx_init_state = torch.cat((ctx_vectors,self.ctx_init_state[n_ctx:self.ctx_init_state.shape[0],:].float()), 0)
        ctx_init_state_temp = torch.cat((self.ctx_init_state[0:2, :].float(), ctx_vectors), 0)
        self.ctx_init_state = torch.cat((ctx_init_state_temp, self.ctx_init_state[3:, :].float()), 0)
        # print(self.ctx_init_state.shape)
        # self.ctx_init_state = torch.cat((ctx_vectors,self.ctx_init_state[n_ctx:self.ctx_init_state.shape[0],:].float()), 0)
        # self.n_ctx = self.ctx_init_state.shape[0]
        # print(newctx_con)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        sim_connection = self.sim_init
        sim_name = []
        for i in range(len(classnames)):
            cls_sim = []
            for j in range(len(classnames)):
                if i == j :
                    new_name = classnames[i]
                else:
                    new_name = classnames[i] + ' '+sim_connection+' '+classnames[j]
                cls_sim.append(new_name)
            sim_name.append(cls_sim)

        ###生成了需要的相似文本


        if not self.learned_cls:
            # classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            sim_name_lens =[]

            for cnames in sim_name:
                # 计算当前类别中每个类名的编码长度
                cname_lens = [len(_tokenizer.encode(name)) for name in cnames]
                # 将计算得到的长度列表添加到 sim_name_lens
                sim_name_lens.append(cname_lens)

            prompts = [self.prompt_prefix + " " + name +"."for name in classnames]

            sim_prompts = []
            for s_name in sim_name:
                s_names = [self.prompt_prefix + " " + s_n+"."for s_n in s_name]
                sim_prompts.append(s_names)



        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim,
                                      dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
            sims = None
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        tokenized_sim_prompts =[]
        for s_prompt in sim_prompts:
            tokenized_prompt = torch.cat([tokenize(p) for p in s_prompt]).to(self.device)
            tokenized_sim_prompts.append(tokenized_prompt)



        clip=self.clip_model

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_sim = []
            for x in tokenized_sim_prompts:
                embedding_sim.append(clip.token_embedding(x).type(self.dtype))

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS


        token_suffix_sim = []
        for x in embedding_sim:
            token_suffix_sim.append(x[:, 1 + self.n_ctx:, :])
        self.token_suffix_sim = token_suffix_sim


        self.name_lens = name_lens
        self.sim_name_lens = sim_name_lens
        self.tokenized_prompts = tokenized_prompts
        self.tokenized_sim_prompts = torch.stack(tokenized_sim_prompts)
        self.classnames = classnames



    def reset_classnames_sim_gen_beifen(self, classnames, arch,sim_matrix = None):
        self.n_cls = len(classnames)
        self.classnames = classnames
        sim_connection = self.sim_init
        self.arch = arch
        similarity_len=[]
        classnames_len=[]
        for i_c in range(len(classnames)):
            classnames_len.append(len(_tokenizer.encode(classnames[i_c])))




        self.classnames_len = classnames_len
        sim_name = []
        for i in range(len(classnames)):
            cls_sim = []
            sim_len = []
            c1_len= []
            c2_len = []
            for j in range(len(classnames)):
                if i == j :
                    new_name = classnames[i]
                    sim_len.append(0)
                    c2_len.append(0)
                else:
                    a = sim_matrix[i][i]
                    b = sim_matrix[i][j]
                    p0, p1 = a / (a + b), b / (a + b)
                    similarity  = "{:.0f}%".format(p1 * 100)
                    new_name = classnames[i] + ' '+similarity+' '+sim_connection+' '+classnames[j]
                    sim_len.append(len(_tokenizer.encode(similarity)))
                cls_sim.append(new_name)
            sim_name.append(cls_sim)
            similarity_len.append(sim_len)

        ###生成了需要的相似文本


        if not self.learned_cls:
            # classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            sim_name_lens =[]

            for cnames in sim_name:
                # 计算当前类别中每个类名的编码长度
                cname_lens = [len(_tokenizer.encode(name)) for name in cnames]
                # 将计算得到的长度列表添加到 sim_name_lens
                sim_name_lens.append(cname_lens)

            prompts = [self.prompt_prefix + " " + name +"."for name in classnames]

            sim_prompts = []
            for s_name in sim_name:
                s_names = [self.prompt_prefix + " " + s_n+"."for s_n in s_name]
                sim_prompts.append(s_names)



        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim,
                                      dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
            sims = None
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        tokenized_sim_prompts =[]
        for s_prompt in sim_prompts:
            tokenized_prompt = torch.cat([tokenize(p) for p in s_prompt]).to(self.device)
            tokenized_sim_prompts.append(tokenized_prompt)



        clip=self.clip_model

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_sim = []
            for x in tokenized_sim_prompts:
                embedding_sim.append(clip.token_embedding(x).type(self.dtype))

        self.token_prefix = embedding[:, :1, :]


        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS

        token_suffix_sim = []
        token_relationship_sim = []
        token_cls1=[]
        token_cls2=[]
        token_similarity = []
        # for x in embedding_sim:
        #     token_cls1.append(x[:,1+self.n_ctx:2+self.n_ctx,:])
        #     token_similarity.append(x[:,2+self.n_ctx:3+self.n_ctx,:])
        #     token_relationship_sim.append(x[:,3+self.n_ctx:3+self.n_ctx+self.n_sim, :])
        #     token_cls2.append(x[:,3+self.n_ctx+self.n_sim:4+self.n_ctx+self.n_sim, :])
        #     token_suffix_sim.append(x[:, 4+self.n_ctx+self.n_sim:, :])


        for ii in range(len(embedding_sim)):
            token_cls1_inner=[]
            token_similarity_inner = []
            token_cls2_inner = []
            token_suffix_sim_inner = []
            for jj in range(embedding_sim[0].shape[0]):
                token_cls1_inner.append(embedding_sim[ii][jj][1+self.n_ctx:1+self.n_ctx+classnames_len[ii],:])
                token_similarity_inner.append(embedding_sim[ii][jj][1+self.n_ctx+classnames_len[ii]:1+self.n_ctx+classnames_len[ii]+similarity_len[ii][jj],:])
                token_cls2_inner.append(embedding_sim[ii][jj][1+self.n_ctx+classnames_len[ii]+similarity_len[ii][jj]+self.n_sim:1+self.n_ctx+classnames_len[ii]+similarity_len[ii][jj]+self.n_sim+classnames_len[jj],:])
                token_suffix_sim_inner.append(embedding_sim[ii][jj][1+self.n_ctx+classnames_len[ii]+similarity_len[ii][jj]+self.n_sim+classnames_len[jj]:,:])

            token_cls1.append(token_cls1_inner)
            token_similarity.append(token_similarity_inner)
            token_cls2.append(token_cls2_inner)
            token_suffix_sim.append(token_suffix_sim_inner)



##################################这个2要改，有的类别可能不是一个单词构成的，但是咋Visda上可以先这么测试########################

        self.token_suffix_sim = token_suffix_sim
        self.token_relationship_sim = token_relationship_sim
        self.token_cls1 = token_cls1
        self.token_cls2 = token_cls2
        self.token_similarity = token_similarity


        self.name_lens = name_lens
        self.sim_name_lens = sim_name_lens
        self.tokenized_prompts = tokenized_prompts
        self.tokenized_sim_prompts = torch.stack(tokenized_sim_prompts)
        self.classnames = classnames

    def reset_classnames_sim_gen(self, classnames, arch, sim_matrix=None):
            self.n_cls = len(classnames)
            self.classnames = classnames
            sim_connection = self.sim_init
            self.arch = arch
            similarity_len = []
            classnames_len = []
            for i_c in range(len(classnames)):
                classnames_len.append(len(_tokenizer.encode(classnames[i_c])))

            self.classnames_len = classnames_len
            sim_name = []
            for i in range(len(classnames)):
                cls_sim = []
                sim_len = []
                c1_len = []
                c2_len = []
                for j in range(len(classnames)):
                    if i == j:
                        new_name = classnames[i]
                        sim_len.append(0)
                        c2_len.append(0)
                    else:
                        new_name = classnames[i] + ' ' + sim_connection + ' ' + classnames[j]
                    cls_sim.append(new_name)
                sim_name.append(cls_sim)

            ###生成了需要的相似文本

            if not self.learned_cls:
                # classnames = [name.replace("_", " ") for name in classnames]
                name_lens = [len(_tokenizer.encode(name)) for name in classnames]
                sim_name_lens = []

                for cnames in sim_name:
                    # 计算当前类别中每个类名的编码长度
                    cname_lens = [len(_tokenizer.encode(name)) for name in cnames]
                    # 将计算得到的长度列表添加到 sim_name_lens
                    sim_name_lens.append(cname_lens)

                prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

                sim_prompts = []
                for s_name in sim_name:
                    s_names = [self.prompt_prefix + " " + s_n + "." for s_n in s_name]
                    sim_prompts.append(s_names)



            else:
                cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim,
                                          dtype=self.dtype)  # assume each learnable cls_token is only 1 word
                nn.init.normal_(cls_vectors, std=0.02)
                cls_token = "X"
                name_lens = [1 for _ in classnames]
                prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
                # TODO: re-init the cls parameters
                # self.cls = nn.Parameter(cls_vectors) # to be optimized
                self.cls_init_state = cls_vectors.detach().clone()
                sims = None
            tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
            tokenized_sim_prompts = []
            for s_prompt in sim_prompts:
                tokenized_prompt = torch.cat([tokenize(p) for p in s_prompt]).to(self.device)
                tokenized_sim_prompts.append(tokenized_prompt)

            clip = self.clip_model

            with torch.no_grad():
                embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)
                embedding_sim = []
                for x in tokenized_sim_prompts:
                    embedding_sim.append(clip.token_embedding(x).type(self.dtype))

            self.token_prefix = embedding[:, :1, :]

            self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS

            token_suffix_sim = []
            token_relationship_sim = []
            token_cls1 = []
            token_cls2 = []
            # token_similarity = []
            # for x in embedding_sim:
            #     token_cls1.append(x[:,1+self.n_ctx:2+self.n_ctx,:])
            #     token_similarity.append(x[:,2+self.n_ctx:3+self.n_ctx,:])
            #     token_relationship_sim.append(x[:,3+self.n_ctx:3+self.n_ctx+self.n_sim, :])
            #     token_cls2.append(x[:,3+self.n_ctx+self.n_sim:4+self.n_ctx+self.n_sim, :])
            #     token_suffix_sim.append(x[:, 4+self.n_ctx+self.n_sim:, :])

            for ii in range(len(embedding_sim)):
                token_cls1_inner = []
                token_cls2_inner = []
                token_suffix_sim_inner = []
                for jj in range(embedding_sim[0].shape[0]):
                    token_cls1_inner.append(
                        embedding_sim[ii][jj][1 + self.n_ctx:1 + self.n_ctx + classnames_len[ii], :])
                    token_cls2_inner.append(embedding_sim[ii][jj][
                                            1 + self.n_ctx + classnames_len[ii]  :1 + self.n_ctx + classnames_len[ii]
                                                                  + self.n_sim + classnames_len[
                                                                     jj], :])
                    token_suffix_sim_inner.append(embedding_sim[ii][jj][
                                                  1 + self.n_ctx + classnames_len[ii] + self.n_sim + classnames_len[jj]:, :])

                token_cls1.append(token_cls1_inner)
                # token_similarity.append(token_similarity_inner)
                token_cls2.append(token_cls2_inner)
                token_suffix_sim.append(token_suffix_sim_inner)

            ##################################这个2要改，有的类别可能不是一个单词构成的，但是咋Visda上可以先这么测试########################

            self.token_suffix_sim = token_suffix_sim
            self.token_relationship_sim = token_relationship_sim
            self.token_cls1 = token_cls1
            self.token_cls2 = token_cls2
            # self.token_similarity = token_similarity

            self.name_lens = name_lens
            self.sim_name_lens = sim_name_lens
            self.tokenized_prompts = tokenized_prompts
            self.tokenized_sim_prompts = torch.stack(tokenized_sim_prompts)
            self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)




        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_sim = self.token_suffix_sim
        if self.fine_gained_sim:
            similarity = self.token_similarity
            cls1 = self.token_cls1
            cls2 = self.token_cls2
            sim = self.sim
            sim = sim.unsqueeze(0).expand( self.n_cls, -1, -1)
        elif self.gained_sim:
            cls1 = self.token_cls1
            cls2 = self.token_cls2
            sim = self.sim
            sim = sim.unsqueeze(0).expand(self.n_cls, -1, -1)

        # if self.fine_gained_sim:
        #
        #     relationship_sim = self.token_relationship_sim
        #     relationship_sim_base = []
        #     for relationships in relationship_sim:
        #         relationship_sim_base.append(self.relationship_model(relationships))
        #     relationship_sim = relationship_sim_base
        #
        # else :
        #     relationship_sim = None

        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:

                if not (self.fine_gained_sim or self.gained_sim):
                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx,  # (n_cls, n_ctx, dim)
                            suffix,  # (n_cls, *, dim)
                        ],
                        dim=-2,
                    )
                    prompts_list = []
                    for x in suffix_sim:
                        prompts_list.append(torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx,  # (n_cls, n_ctx, dim)
                            x,  # (n_cls, *, dim)
                        ],
                        dim=-2,
                    ))
                    self.prompts_list = torch.stack(prompts_list)
                elif self.fine_gained_sim:
                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx,  # (n_cls, n_ctx, dim)
                            suffix,  # (n_cls, *, dim)
                        ],
                        dim=-2,
                    )
                    prompts_list = []
                    # for a,b,c,d in zip(cls1,similarity,cls2,suffix_sim):
                    #     prompts_list.append(torch.cat(
                    #         [
                    #             prefix,  # (n_cls, 1, dim)
                    #             ctx,  # (n_cls, n_ctx, dim)
                    #             a,    #(n_cls, n_sim, dim)
                    #             b,  # (n_cls, *, dim)
                    #             sim,
                    #             c,
                    #             d
                    #         ],
                    #         dim=-2,
                    #     ))
                    for  ii in range (len(cls1)):
                        p_inner = []
                        for jj in range(len(cls1)):
                            p_inner.append(torch.cat([
                                prefix[0],
                                ctx[0],
                                cls1[ii][jj],
                                similarity[ii][jj],
                                sim[0],
                                cls2[ii][jj],
                                suffix_sim[ii][jj]

                            ],
                            dim = -2
                            ),
                            )
                        prompts_list.append(torch.stack(p_inner))

                    self.prompts_list = torch.stack(prompts_list)
                elif self.gained_sim:
                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx,  # (n_cls, n_ctx, dim)
                            suffix,  # (n_cls, *, dim)
                        ],
                        dim=-2,
                    )
                    prompts_list = []
                    for  ii in range (len(cls1)):
                        p_inner = []
                        for jj in range(len(cls1)):
                            p_inner.append(torch.cat([
                                prefix[0],
                                ctx[0],
                                cls1[ii][jj],
                                cls2[ii][jj],
                                suffix_sim[ii][jj]

                            ],
                            dim = -2
                            ),
                            )
                        prompts_list.append(torch.stack(p_inner))
                    self.prompts_list = torch.stack(prompts_list)





        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None,ctx_position='end', learned_cls=False,logit_rate =None,fine_gained_sim = False,gained_sim = False,sim_matrix = None,sim_init = "looks like a",visual_prompt = False):
        super(ClipTestTimeTuning, self).__init__()
        self.sim_indices = None
        self.text_sim_features = None
        self.text_features = None

        if visual_prompt:
            clip = load_clip_to_cpu(arch)
            clip = clip.cuda()
        else:
            clip, _= load(arch, device="cuda")#重点地方，应该对这个进行更改

        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.sim_matrix = sim_matrix
        # self.logit_scale = clip.logit_scale.data
        if logit_rate is None:
            self.logit_scale = clip.logit_scale.data
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_rate))

        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls,fine_gained_sim=fine_gained_sim,gained_sim=gained_sim,sim_matrix=sim_matrix,sim_init=sim_init)
        self.criterion = criterion
        self.text_features_wd = None
        self.clip_model = clip

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def reset_classnames_sim_gen(self,classnames,arch,sim_matrix = None):
        self.prompt_learner.reset_classnames_sim_gen(classnames, arch,sim_matrix)
    def reset_classnames_sim_gen_beifen(self,classnames,arch,sim_matrix = None):
        self.prompt_learner.reset_classnames_sim_gen_beifen(classnames, arch,sim_matrix)

    def reset_prompt(self, args, ent_interval, adv_dic, pesu_label):
        self.prompt_learner.reset_prompt(args, ent_interval, adv_dic, pesu_label)

    def get_text_features(self):

        if self.sim_indices is None:
            self.modify_sim_text()
        sim_indices = self.sim_indices
        prompts = self.prompt_learner()
        prompts_sim_list = self.prompt_learner.prompts_list
        for ii in range(prompts.shape[0]):
            for jj in range(prompts.shape[0]):
                if ii ==jj:
                    prompts_sim_list[ii][jj]=prompts[ii]
        prompts_sim_list = prompts_sim_list[sim_indices[:,0], sim_indices[:,1]]


        tokenized_prompts = self.prompt_learner.tokenized_prompts
        tokenized_sim_prompts = self.prompt_learner.tokenized_sim_prompts
        tokenized_sim_prompts = tokenized_sim_prompts[sim_indices[:,0], sim_indices[:,1]]


        self.text_sim_features = self.text_encoder(prompts_sim_list, tokenized_sim_prompts)
        self.text_sim_features = self.text_sim_features/self.text_sim_features.norm(dim=-1,keepdim=True)

        self.text_features = self.text_encoder(prompts,tokenized_prompts)
        self.text_features  = self.text_features/self.text_features.norm(dim=-1,keepdim= True)

        text_features_indices = utils.get_pure_text_features_idx(sim_indices,self.prompt_learner.n_cls)
        # text_features = text_sim_features[text_features_indices]







        return self.text_features


    def modify_sim_prompt(self,exist_indices=None,sim_matrix = None):
        if exist_indices is not None:
            self.sim_indices = exist_indices.cuda()
            self.sim_matrix = sim_matrix
            return
        with torch.no_grad():
            text_features = []
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            t_features = self.text_encoder(prompts, tokenized_prompts)
            text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
            text_features = torch.stack(text_features, dim=0)
            text_features = torch.mean(text_features, dim=0)

            t_f_mat = text_features@text_features.T
            n_cls = text_features.shape[0]
            top_k = n_cls*4
            t_f_mat = t_f_mat.view(-1)
            values, indices = t_f_mat.topk(top_k, largest=True)
            indices_x ,indices_y = indices//n_cls,indices%n_cls

            sim_indices = torch.stack((indices_x, indices_y), dim=1)
            self.sim_indices=sim_indices.long().cuda()

    def modify_sim_text(self, exist_indices=None, sim_matrix=None):
        if exist_indices is not None:
            self.sim_indices = exist_indices.cuda()
            return
        with torch.no_grad():
            text_features = []
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            t_features = self.text_encoder(prompts, tokenized_prompts)
            text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
            text_features = torch.stack(text_features, dim=0)
            text_features = torch.mean(text_features, dim=0)

            t_f_mat = text_features @ text_features.T
            n_cls = text_features.shape[0]
            top_k = n_cls * 4
            t_f_mat = t_f_mat.view(-1)
            values, indices = t_f_mat.topk(top_k, largest=True)
            indices_x, indices_y = indices // n_cls, indices % n_cls

            sim_indices = torch.stack((indices_x, indices_y), dim=1)
            self.sim_indices = sim_indices.long().cuda()





    def pretrain_sim_connection(self,prob_center = None,batch_size = 16):

        s = torch.randint(10, 45, (batch_size,))
        flip = torch.randint(0,2,(batch_size,))
        flip = flip.bool()
        sim_connection = self.prompt_learner.sim_init
        classnames = self.prompt_learner.classnames
        classnames_len = self.prompt_learner.classnames_len
        if prob_center is None:
            a = torch.randint(0, len(classnames)-1, (batch_size, 2))
        else :
            prob_clone = prob_center.clone().detach()
            prob_clone.fill_diagonal_(0)
            t_f_mat = prob_clone.view(-1)
            t_f_mat_mean, t_f_mat_var = t_f_mat.mean(), t_f_mat.var()
            top_k = (t_f_mat >= (t_f_mat_mean + t_f_mat_var)).sum()
            top_k = min(top_k,batch_size)
            values, indices = t_f_mat.topk(top_k, largest=True)
            indices_x, indices_y = indices // prob_clone.shape[0], indices % prob_clone.shape[0]
            sim_indices = torch.stack((indices_x, indices_y), dim=1)
            indices_z = torch.randperm(sim_indices.shape[0])[:batch_size]
            batch_size = min(batch_size,sim_indices.shape[0])
            a = sim_indices[indices_z]
            flip = flip[:batch_size]
            a[flip,0],a[flip,1]=a[flip,1],a[flip,0]


        sim_sentence = []
        for i in range(batch_size):
            if a[i][0]== a[i][1]:
                a[i][1] = (a[i][1]+random.randint(1,len(classnames)-1))%(len(classnames)-1)
            similarity = f"{s[i]}%"
            new_name = self.prompt_learner.prompt_prefix+' '+classnames[a[i][0]]+' '+similarity+' '+sim_connection+' '+classnames[a[i][1]]
            sim_sentence.append(new_name)
        similarity_len = torch.ceil(torch.log10(s)).to(torch.int32)


        tokenized_prompts = torch.cat([tokenize(p) for p in sim_sentence]).to(self.prompt_learner.device)
        clip=self.clip_model
        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.prompt_learner.dtype)

        token_cls1= []
        token_similarity =[]
        token_cls2 = []
        token_suffix= []
        token_prefix = embedding[:, :1, :]
        for iii in range(embedding.shape[0]):
            token_cls1.append(embedding[iii][1+self.prompt_learner.n_ctx:1+self.prompt_learner.n_ctx+classnames_len[a[iii][0]]])
            token_similarity.append(embedding[iii][1+self.prompt_learner.n_ctx+classnames_len[a[iii][0]] : 1+self.prompt_learner.n_ctx+classnames_len[a[iii][0]]+similarity_len[iii]])

            token_cls2.append(embedding[iii][1+self.prompt_learner.n_ctx+classnames_len[a[iii][0]]+similarity_len[iii]+self.prompt_learner.n_sim : 1+self.prompt_learner.n_ctx+classnames_len[a[iii][0]]+similarity_len[iii]+self.prompt_learner.n_sim+ classnames_len[a[iii][1]]])
            token_suffix.append(embedding[iii][1+self.prompt_learner.n_ctx+classnames_len[a[iii][0]]+similarity_len[iii]+self.prompt_learner.n_sim+ classnames_len[a[iii][1]]:])

        # token_cls1 = embedding[:, 1+self.prompt_learner.n_ctx:2+self.prompt_learner.n_ctx, :]
        # token_similarity = embedding[:, 2+self.prompt_learner.n_ctx:3+self.prompt_learner.n_ctx, :]
        # token_relationship = embedding[:, 3+self.prompt_learner.n_ctx:3+self.prompt_learner.n_ctx+self.prompt_learner.n_sim, :]
        # token_cls2 = embedding[:, 3+self.prompt_learner.n_ctx+self.prompt_learner.n_sim:4+self.prompt_learner.n_ctx+self.prompt_learner.n_sim, :]
        # token_suffix = embedding[:, 4+self.prompt_learner.n_ctx+self.prompt_learner.n_sim:, :]


        ctx = self.prompt_learner.ctx
        sim = self.prompt_learner.sim


        prompts_l = []
        for ii in range (batch_size):
            prompts_l.append(torch.cat([
                token_prefix[0],
                ctx,
                token_cls1[ii],
                token_similarity[ii],
                sim,
                token_cls2[ii],
                token_suffix[ii]
            ],
            dim=-2
            ),
            )
        prompts = torch.stack(prompts_l)

        # prompts = torch.cat(
        #     [
        #         token_prefix,
        #         ctx,
        #         token_cls1,
        #         token_similarity,
        #         sim,
        #         token_cls2,
        #         token_suffix
        #     ],
        #     dim=-2,
        # )

        pretrain_text_features = self.text_encoder(prompts,tokenized_prompts)
        pretrain_text_features = pretrain_text_features/pretrain_text_features.norm(dim=-1,keepdim=True)

        if self.text_features == None:
            text_features = self.get_text_features()
        else:
            text_features = self.text_features
        logit_scale = self.logit_scale.exp()
        logit_scale = logit_scale.detach()
        text_features = text_features.detach()

        pretrain_prob = logit_scale*pretrain_text_features@text_features.T
        pretrain_prob = pretrain_prob.softmax(dim=1)

        loss_fuc = torch.nn.MSELoss()
        template_tensor = torch.zeros_like(pretrain_prob)
        eff_tensor = torch.ones(pretrain_prob.shape[0])
        for i in range(batch_size):
            p1 = s[i]/100
            p0 = 1-p1

            # if p0>0.8:
            #     p0 = 0.8
            #     p1 = 0.2

            template_tensor[i][a[i][0]] = p0
            template_tensor[i][a[i][1]] = p1
            eff_tensor[i] = 1


        return loss_fuc(pretrain_prob,template_tensor.detach())

    def inference(self, image,image_features=None,need_new_text_features=False):
        if image_features is None:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))

        #############################这儿有问题
        if self.text_features==None or need_new_text_features:
            text_features = self.get_text_features()
        else :
            text_features = self.text_features
        # text_features_save = text_features.clone().detach()
        # torch.save(text_features_save, 'learned_features.pt')
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits.softmax(dim=1) ,image_features


    def inference_float32(self, image,image_features=None,need_new_text_features=False):
        if image_features is None:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))

        #############################这儿有问题
        if self.text_features==None or need_new_text_features:
            text_features = self.get_text_features()
        else :
            text_features = self.text_features
        # text_features_save = text_features.clone().detach()
        # torch.save(text_features_save, 'learned_features.pt')
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits.softmax(dim=1).to(torch.float32), image_features



    def inference_with_updated_logits(self, logits_new):
        # with torch.no_grad():
        # image_features = self.image_encoder(image.type(self.dtype))
        # image_features = all_clip_feature_prompt
        text_features = self.get_text_features()
        # text_features_save = text_features.clone().detach()
        # torch.save(text_features_save, 'learned_features.pt')
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()

        return logits_new, text_features

    def inference_by_sim(self,image,image_features = None,T = None):
        if image_features is None:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))

            #############################这儿有问题
        if self.text_sim_features == None:
            _ =self.get_text_features()
            text_sim_features = self.text_sim_features

        else:
            text_sim_features = self.text_sim_features
        logit_scale = self.logit_scale.exp()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prob = logit_scale * image_features @ text_sim_features.t()
        idx = prob.argmax(dim=1)
        return self.sim_indices[idx]


    def get_image_features0(self,image):
        image_features = self.image_encoder(image.type(self.dtype))
        return image_features

    def inference_by_sim_probs(self,image,image_features = None,need_new_text_features=False,need_softmax =True):
        if image_features is None:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))
        else:
            image_features = image_features.detach()
        if self.text_features == None or need_new_text_features:
            _ = self.get_text_features()
            text_sim_features = self.text_sim_features
        else:
            text_sim_features = self.text_sim_features
        logit_scale = self.logit_scale.exp()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prob_hard = logit_scale * image_features @ text_sim_features.t()

        prob = torch.zeros(image_features.shape[0],self.text_features.shape[0]).cuda()


        for i in range(self.text_features.shape[0]):
            mask = self.sim_indices[:,0]==i
            # 使用bool_tensor作为索引选择第二维上的元素
            selected_prob = prob_hard[:, mask]
            selected_prob = selected_prob.max(dim=1)[0]
            prob[:,i]=selected_prob
        if need_softmax:
            # return prob.softmax(dim=1)
            return utils.softmax_with_temperature(prob,temperature=1)
        else:
            return prob


    def inference_by_sim_probs_return_fea(self,image,image_features = None,need_new_text_features=False,need_softmax =True):
        if image_features is None:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))
        if self.text_features == None or need_new_text_features:
            t = self.get_text_features()
            text_sim_features = self.text_sim_features
        else:
            text_sim_features = self.text_sim_features
        logit_scale = self.logit_scale.exp()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prob_hard = logit_scale * image_features @ text_sim_features.t()

        prob = torch.zeros(image_features.shape[0],self.text_features.shape[0]).cuda()
        text_fea = self.text_features


        for i in range(self.text_features.shape[0]):
            mask = self.sim_indices[:,0]==i
            # 使用bool_tensor作为索引选择第二维上的元素
            selected_prob = prob_hard[:, mask]
            selected_prob = selected_prob.max(dim=1)[0]
            prob[:,i]=selected_prob
        if need_softmax:
            # return prob.softmax(dim=1)
            return utils.softmax_with_temperature(prob,temperature=1),image_features,text_fea
        else:
            return prob,image_features,text_fea





    def inference_by_sim_probs_and_indices(self,image,image_features = None,need_new_text_features=False):
        if image_features is None:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))
        if self.text_features == None or need_new_text_features:
            _ = self.get_text_features()
            text_sim_features = self.text_sim_features
        else:
            text_sim_features = self.text_sim_features
        logit_scale = self.logit_scale.exp()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prob_hard = logit_scale * image_features @ text_sim_features.t()
        prob = torch.zeros(image_features.shape[0],self.text_features.shape[0]).cuda()
        idx = prob_hard.argmax(dim=1)
        sim_indices_idx = self.sim_indices[idx]

        for i in range(self.text_features.shape[0]):
            mask = self.sim_indices[:,0]==i
            # 使用bool_tensor作为索引选择第二维上的元素
            selected_prob = prob_hard[:, mask]
            selected_prob = selected_prob.max(dim=1)[0]
            prob[:,i]=selected_prob



        return prob.softmax(dim=1),sim_indices_idx





    def calculate_sim_similarity(self):
        _=self.get_text_features()
        logit_scale = self.logit_scale.exp()
        text_features = self.text_features
        text_sim_features = self.text_sim_features
        hard_prob=logit_scale * text_sim_features @ text_features.T

        return torch.softmax(hard_prob,dim=1)




    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False,logit_rate = None,fine_gained_sim=False,gained_sim = False,sim_init = "looks like a",visual_prompt = False):
    if test_set == "VISDA-C":
        classnames = ['plane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
                           'skateboard', 'train', "truck"]
    elif test_set == 'office-home':
        classnames = [
            "alarm clock", "backpack", "batteries", "bed", "bike",
            "bottle", "bucket", "calculator", "calendar", "candles",
            "chair", "clipboards", "computer", "couch", "curtains",
            "desk lamp", "drill", "eraser", "exit sign", "fan",
            "file cabinet", "flipflops", "flowers", "folder", "fork",
            "glasses", "hammer", "helmet", "kettle", "keyboard",
            "knives", "lamp shade", "laptop", "marker", "monitor",
            "mop", "mouse", "mug", "notebook", "oven",
            "pan", "paper clip", "pen", "pencil", "postit notes",
            "printer", "push pin", "radio", "refrigerator", "ruler",
            "scissors", "screwdriver", "shelf", "sink", "sneakers",
            "soda", "speaker", "spoon", "table", "telephone",
            "toothbrush", "toys", "trash can", "tv", "webcam"
            ]
    elif test_set =="DomainNet126":
        classnames = []
        with open("/home/q23301278/Codes/COVA-JMDS/data/DomainNet126/classname.txt", 'r') as file:
            # 使用readlines()方法读取文件的所有行
            lines = file.readlines()

            # 遍历每一行，strip()方法用于移除末尾的换行符
            for line in lines:
                classnames.append(line.strip())


    elif test_set =='office-31':
        classnames = []
        with open("/home/q23301278/Codes/COVA-JMDS/data/office-31/classname.txt", 'r') as file:
            # 使用readlines()方法读取文件的所有行
            lines = file.readlines()

            # 遍历每一行，strip()方法用于移除末尾的换行符
            for line in lines:
                classnames.append(line.strip())

    # if fine_gained_sim:
    #   # sim_init = "chance it will be mistaken for a"

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls,ctx_position='end',logit_rate=logit_rate,fine_gained_sim=fine_gained_sim,gained_sim=gained_sim,sim_init=sim_init,visual_prompt=visual_prompt)

    return model


def load_clip_to_cpu(backbone_name, zero_shot_model=False):
    # url = clip._MODELS[backbone_name]


    if backbone_name == "ViT-B/16":
        model_path = '/ViT-B-16.pt'
    elif backbone_name == "ViT-L/14":
        model_path = '/ViT-L-14.pt'
    elif backbone_name == "ViT-B/32":
        model_path = '/ViT-B-32.pt'
    else:
        print('enter the wrong  name.')

    root = os.path.expanduser("~/.cache/clip")
    model_path = root+model_path

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": 1,
                      "language_depth": 0,
                      "vision_ctx": 4,
                      "language_ctx": 4}
    model = clip_models.clip.build_model(state_dict or model.state_dict(), design_details)

    return model





