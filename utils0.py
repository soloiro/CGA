
import torch.nn.functional as F

import torch.nn as nn

from einops import rearrange
import torch

def get_class_centroids_TOP2_v2(prob,mean_p = 0.5,var_p = 0.1,normalize = True):#使用统计得到的数据知识对threshold进行计算
    N, D = prob.shape  # N是logits的batch_size D是分类数
    K = D  # K是分类数

    cl = prob.argsort(dim=1, descending=False)



    cl1 = cl[:, -1]
    cl2 = cl[:, -2]
    cl1_weight = torch.gather(prob, 1, cl1.view(-1, 1)).squeeze()
    cl2_weight = torch.gather(prob, 1, cl2.view(-1, 1)).squeeze()
    # cl1_weight[cl1_weight > (mean_p-2*var_p)] = 1
    # cl2_weight[cl2_weight > threshold] = 1

    prob_weight1 = prob * (cl1_weight.unsqueeze(1))
    prob_weight2 = prob * (cl2_weight.unsqueeze(1))
    cl1 = cl1.long().view(-1)  # -> class index  cl是prob的每个样本中最小的最不可能的序数，然后转换为一维
    cl2 = cl2.long().view(-1)
    Ncl1 = cl1.view(cl1.size(0), 1).expand(-1, D)  # 先将cl的shape从N变为N,1，在变成N,K其中元素就cl的扩展，每个N中是全部一样的,赋给Ncl
    Ncl2 = cl2.view(cl2.size(0), 1).expand(-1, D)
    labels_count_all_weight = torch.zeros([K]).cuda()
    for i in range(K):
        class_sum1 =(cl1_weight[cl1 == i]).sum()
        class_sum2 = (cl2_weight[cl2 == i]).sum()
        if class_sum1 == 0 and class_sum2 == 0:
            labels_count_all_weight[i] = 1
        else:
            labels_count_all_weight[i] = class_sum1+class_sum2
    Ncl =torch.cat((Ncl1,Ncl2),0)
    prob_all = torch.cat((prob_weight1,prob_weight2),0)
    c = torch.zeros([K, D], dtype=prob.dtype).cuda().scatter_add_(0, Ncl, prob_all)  # -> class centroids
    if normalize:
        c = torch.nn.functional.normalize(c,p=1) # 归一化
    # c = c / labels_count_all_weight.float().unsqueeze(1)  # 计算出一个按照按照最小的logits聚类后相加的结果，再平均的一个
    return cl1, c

class ContrastiveLoss1(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, F, Fa):
        """
        F: 原始特征，形状 [batch_size, fea_dim]
        Fa: 增强后的特征，形状 [batch_size, fea_dim]
        """
        # 特征归一化（确保余弦相似度计算正确）
        F = F / F.norm(dim=1, keepdim=True)
        Fa = Fa / Fa.norm(dim=1, keepdim=True)

        # 计算相似度矩阵（余弦相似度）
        sim_matrix = torch.mm(F, Fa.T)  # [batch_size, batch_size]
        sim_matrix /= self.temperature

        # 标签为对角线位置（每个样本i的正样本是Fa[i]）
        labels = torch.arange(F.size(0), device=F.device)  # [0, 1, 2, ..., batch_size-1]

        # 计算交叉熵损失
        loss = self.criterion(sim_matrix, labels)
        return loss

def refine_x_outputs(x:torch.Tensor,x_c:torch.Tensor,iter_rate=0.5,target_par_label = None,selected_indices=None):
    x_average = torch.ones_like(x[0])
    x_average = x_average/x_average.shape[0]
    max_entropy = (- x_average * torch.log(x_average + 1e-6)).sum(dim=0)

    x_confidence = torch.zeros(x.shape[0])
    x_refined = torch.zeros_like(x)
    x_argmax = torch.argmax(x,dim =1)
    x_c_argmax =  torch.argmax(x_c,dim =1)
    same_top1 = x_argmax==x_c_argmax
    ii =0

    for i in range(same_top1.shape[0]):
        if same_top1[i]:
            x_refined[i]= 0.99*x[i]+0.01*x_c[i] if max(x[i])>max(x_c[i]) else 0.99*x_c[i]+0.01*x[i]
            x_confidence[i] = max(max_entropy-(- x_c[i] * torch.log(x_c[i] + 1e-6)).sum(dim=0),max_entropy-(- x[i] * torch.log(x[i] + 1e-6)).sum(dim=0),1)
        else:
            if target_par_label is not None and selected_indices[i]:
                x_i=x[i] * (0.5)+target_par_label[i]*0.5
            else :
                x_i = x[i]
            x_ent=(- x_i * torch.log(x_i + 1e-6)).sum(dim=0)
            x_c_ent = (- x_c[i] * torch.log(x_c[i] + 1e-6)).sum(dim=0)

            if x_ent<=0.01 and x_c_ent<=0.01:
                p = 0.5
                x_ent = 0.01
                x_c_ent = 0.01
            elif x_ent >0.01:
                x_c_ent = 0.01
                p = x_ent / (x_ent + x_c_ent)
            elif x_c_ent >0.01:
                x_ent = 0.01
                p = x_ent / (x_ent + x_c_ent)
            else:
                p = x_ent/(x_ent+x_c_ent)

            x_refined[i] = (1 - p) * x_i + p * x_c[i]

            x_confidence[i] =max(max_entropy-(- x_refined[i] * torch.log(x_refined[i] + 1e-6)).sum(dim=0)-(iter_rate*0.6),0)
    return x_refined,(x_confidence.cuda()/max_entropy).detach()

class IFT_Module_fea(nn.Module):
    """ IFT """

    def __init__(self, fea_dim, beta_s=1.0, beta_t=1.0,
                 ):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        input_dim = fea_dim
        pre_dim1 = input_dim // 8
        pre_dim2 = input_dim // 8

        self.beta_s = beta_s
        self.beta_t = beta_t
        self.scale = 0.1



        self.pre_project = nn.Sequential(  # 3 layers
            nn.Linear(input_dim, pre_dim1),
            nn.BatchNorm1d(pre_dim1),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim1, pre_dim2),
            nn.BatchNorm1d(pre_dim2),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim2, input_dim * 3)
        )#.half()

        self.post_project = nn.Sequential(  # only one layer
            nn.Linear(input_dim, input_dim)
        )#.half()

        self.dynamic_weight = nn.Sequential(
            nn.Linear(input_dim, pre_dim1),
            nn.ReLU(),
            nn.Linear(pre_dim1, 1),
            nn.Sigmoid()
        )




    def forward(self, Fv,Fvs_bank,Fvt_bank):
        '''
        Fvs with shape (batch, C): source visual output w/o attnpool
        Fvt with shape (N, C): classes of target visual output w/o attnpool
        '''
        out_fv = self.pre_project(Fv)  # (batch, 3 * C)
        out_fvs = self.pre_project(Fvs_bank)  # (N, 3 * C)
        out_fvt = self.pre_project(Fvt_bank)  # (N, 3 * C)

        q_fv, k_fv, v_fv = tuple(rearrange(out_fv, 'b (d k) -> k b d ', k=3))
        q_fvs, k_fvs, v_fvs = tuple(rearrange(out_fvs, 'b (d k) -> k b d ', k=3))
        q_fvt, k_fvt, v_fvt = tuple(rearrange(out_fvt, 'b (d k) -> k b d ', k=3))

        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))  # (batch, N)
        At = self.softmax(self.scale * q_fv @ k_fvt.permute(1, 0))  # (batch, N)

        Fsp = self.post_project(As @ v_fvs)
        Ftp = self.post_project(At @ v_fvt)

        Fsa = Fv + Fsp  # (batch, C)
        Fta = Fv + Ftp  # (batch, C)

        Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)
        Fta = Fta / Fta.norm(dim=-1, keepdim=True)

        alpha = self.dynamic_weight(Fv)
        Fa = alpha*Fsa+(1-alpha)*Fta

        return Fa,Fsa,Fta,Fsp,Ftp