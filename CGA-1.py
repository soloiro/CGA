import argparse
import os, sys
import os.path as osp
import torchvision
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torchvision import transforms

import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
import time

##############将几个KL改成了CELoss,mixup中将refine后的confidence得到了进行计算 prompt中改变了refine的方式以及KLLOSS###################
############将clip_sim的输出变成概率并且prompt这个sim的词向量概率#######
#################topk最小值，######################

matplotlib.use('Agg')

def fuse_fea(all_fea,index,weight = None):
    if weight is  None:
        fea_set = torch.zeros(index.shape[0],all_fea.shape[1])
        for i in range(index.shape[0]):
            fea_set[i] =all_fea[index[i]].mean(dim=0)
    else :
        fea_set = torch.zeros(index.shape[0], all_fea.shape[1])
        weight = weight.cuda()
        for i in range(index.shape[0]):
            all_fea_set,weight_set = all_fea[index[i]],weight[i]
            # weight_set = weight_set.unsqueeze(-1)
            # weighted_features = all_fea_set * weight_set
            # weighted_features = torch.sum(weighted_features, dim=0)
            # total_weight = torch.sum(weight_set)
            # eps = torch.finfo(weighted_features.dtype).eps  # 自动获取数值精度对应的极小值
            # fea_set[i] = weighted_features / (total_weight + eps)
            fea_set[i] = (all_fea_set * weight_set.unsqueeze(-1)).sum(0) / (weight_set.sum() + 1e-8)

    return fea_set





def get_confused_index(class_a,class_b,prob):
    if class_b is None:
        return prob[:,class_a]
    sum =  prob[:,class_a]+prob[:,class_b]
    dis =  prob[:,class_a]-prob[:,class_b]
    dis = torch.abs(dis)
    return sum+dis

def get_confused_index_new(class_a,class_b,prob,matrix):
    if class_b is None:
        return prob[:,class_a]
    alpha = matrix[class_a][class_a]/matrix[class_a][class_b]
    sum =  prob[:,class_a]+prob[:,class_b]
    dis =  prob[:,class_a]/prob[:,class_b]
    dis1 = dis-alpha
    dis1 =torch.abs(dis1)
    dis1 = dis1/(dis+alpha)

    return sum+dis1


def get_top_index(x,N):
    values, indices = torch.topk(x, k=N)
    return indices

def get_selected_sample_index(prob,sim_indices,cls_num,prob_center,K = 0.5):
    each_cls_num = int(prob.shape[0]*K/cls_num)
    confused_cls_index = torch.zeros(sim_indices.shape[0],each_cls_num)
    confused_cls_index = confused_cls_index.to(torch.long)
    weight_value = torch.empty(sim_indices.shape[0],each_cls_num)
    weight_value = weight_value.to(torch.float)
    weight_value = weight_value.cuda()
    # weight_index = torch.zeros_like(confused_cls_index)
    # weight_index = weight_index.to(torch.float)
    for i in range(sim_indices.shape[0]):
        if sim_indices[i][0]==sim_indices[i][1]:
            class_a = sim_indices[i][0]
            confused_index = get_confused_index(class_a,None,prob)
            confused_cls_index[i] = get_top_index(confused_index,each_cls_num)
            weight_value[i] = confused_index[confused_cls_index[i]]

        else :
            class_a = sim_indices[i][0]
            class_b = sim_indices[i][1]
            confused_index = get_confused_index_new(class_a, class_b, prob,prob_center)

            confused_cls_index[i] = get_top_index(confused_index,each_cls_num)
            weight_value[i] = confused_index[confused_cls_index[i]]


    return confused_cls_index.type(torch.long),weight_value


def update_clip_outputs(loader, clip_model):
    print("clip_model_calculate:")
    with torch.no_grad():
        start_test = True
        iter_test = iter(loader)
        for _ in tqdm(range(len(loader))):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            if start_test:
                new_flag = True
            else:
                new_flag = False

            outputs_clip, images_fea = (clip_model.inference(inputs))
            outputs_clip = torch.softmax(outputs_clip, dim=1)
            outputs_clip = outputs_clip.cuda()
            outputs_clip_sim = (clip_model.inference_by_sim_probs(inputs, images_fea,need_new_text_features=new_flag)).cuda()

            if start_test:
                all_labels = labels.float()
                all_outputs_clip = outputs_clip.float()
                all_outputs_clip_sim = outputs_clip_sim.float()
                start_test = False
            else:
                all_outputs_clip = torch.cat((all_outputs_clip, outputs_clip.float()), 0)
                all_labels = torch.cat((all_labels, labels.float()), 0)
                all_outputs_clip_sim = torch.cat((all_outputs_clip_sim, outputs_clip_sim.float()), 0)

        indices = torch.arange(all_outputs_clip.shape[1]).cuda()

        # 使用 unsqueeze() 在第二维添加一个维度
        indices_unsqueezed = indices.unsqueeze(1)

        # 使用 expand() 复制第二维以匹配第一维的大小
        target_tensor = indices_unsqueezed.expand(-1, 2)


        n_cls = all_outputs_clip.shape[1]
        # top_k = 3*n_cls
        TOP12_connection = \
            utils.get_class_centroids_TOP2_v2(all_outputs_clip, normalize=True)[1]
        TOP12_connection0 = TOP12_connection.clone().detach()
        t_f_mat = TOP12_connection.view(-1)
        t_f_mat_mean, t_f_mat_var = t_f_mat.mean(), t_f_mat.var()
        TOP12_connection.fill_diagonal_(0)
        t_f_mat = TOP12_connection.view(-1)

        top_k = (t_f_mat >= (t_f_mat_mean + t_f_mat_var)).sum()
        top_k = max(top_k, int(n_cls * 1))

        values, indices = t_f_mat.topk(top_k, largest=True)
        indices_x, indices_y = torch.div(indices,n_cls,rounding_mode='floor') , indices % n_cls

        sim_indices = torch.stack((indices_x, indices_y), dim=1)
        sim_indices = torch.cat((sim_indices, target_tensor), dim=0)
        selected_sample_index,weight_set = get_selected_sample_index(all_outputs_clip, sim_indices, all_outputs_clip.shape[1],TOP12_connection0)





    return all_outputs_clip, all_outputs_clip_sim,selected_sample_index,weight_set

def Refine_sim_Loss_fine_gain(clip_model,prob_center):
    sim_prob = clip_model.calculate_sim_similarity()
    sim_prob = sim_prob.cuda()
    sim_idx = clip_model.sim_indices

    loss_fuc = torch.nn.MSELoss()
    template_tensor = torch.zeros_like(sim_prob)
    eff_tensor = torch.ones(sim_prob.shape[0])
    for i in range(sim_prob.shape[0]):
        if sim_idx[i][0] == sim_idx[i][1]:
            template_tensor[i][sim_idx[i][0]] = 1
            eff_tensor[i] = 1
        else:
            a = prob_center[sim_idx[i][0]][sim_idx[i][0]]
            b = prob_center[sim_idx[i][0]][sim_idx[i][1]]
            p0,p1 =a/(a+b),b/(a+b)
            if p0<p1:
                p0 = 0.6
                p1 = 0.4
            template_tensor[i][sim_idx[i][0]] = p0
            template_tensor[i][sim_idx[i][1]] = p1
            eff_tensor[i] = 1
    return loss_fuc(sim_prob,template_tensor)

def refine_x_outputs(x: torch.Tensor, x_c: torch.Tensor, iter_rate=0.5):
    x_average = torch.ones_like(x[0])
    x_average = x_average / x_average.shape[0]
    max_entropy = (- x_average * torch.log(x_average + 1e-6)).sum(dim=0)

    x_confidence = torch.zeros(x.shape[0])
    x_refined = torch.zeros_like(x)
    x_argmax = torch.argmax(x, dim=1)
    x_c_argmax = torch.argmax(x_c, dim=1)
    same_top1 = x_argmax == x_c_argmax
    for i in range(same_top1.shape[0]):
        if same_top1[i]:
            x_refined[i] = 0.9 * x[i] + 0.1 * x_c[i] if max(x[i]) > max(x_c[i]) else 0.9 * x_c[i] + 0.1 * x[i]
            x_confidence[i] = max(max_entropy - (- x_c[i] * torch.log(x_c[i] + 1e-6)).sum(dim=0),
                                  max_entropy - (- x[i] * torch.log(x[i] + 1e-6)).sum(dim=0), 1)
        else:
            x_ent = (- x[i] * torch.log(x[i] + 1e-6)).sum(dim=0)
            x_c_ent = (- x_c[i] * torch.log(x_c[i] + 1e-6)).sum(dim=0)
            if x_ent < 0:
                x_ent = 0.01
            if x_c_ent < 0:
                x_c_ent = 0.01
            p = x_ent / (x_ent + x_c_ent)
            x_refined[i] = (1 - p) * x[i] + p * x_c[i]
            x_confidence[i] = max(max_entropy - (- x_refined[i] * torch.log(x_refined[i] + 1e-6)).sum(dim=0), 0)
    return x_refined, (x_confidence.cuda() / max_entropy).detach()


def Refine_sim_Loss(clip_model):
    sim_prob = clip_model.calculate_sim_similarity()
    sim_prob = sim_prob.cuda()
    sim_idx = clip_model.sim_indices
    criterion_ce = torch.nn.CrossEntropyLoss()
    # mle_loss = 0
    template_tensor = torch.zeros_like(sim_prob)
    eff_tensor = torch.ones(sim_prob.shape[0])
    for i in range(sim_prob.shape[0]):
        if sim_idx[i][0] == sim_idx[i][1]:
            template_tensor[i][sim_idx[i][0]] = 1
            eff_tensor[i] = 1
        else:
            template_tensor[i][sim_idx[i][0]] = 0.8
            template_tensor[i][sim_idx[i][1]] = 0.2
            eff_tensor[i] = 1###change

    return NewKLLoss(sim_prob, template_tensor, eff_tensor.cuda(), eff_tensor)


def prompt_adjust(images1, x_outputs, clip_model, iter_num, max_iter, interval_iter, prob_center,args):
    coefficient = 0.01
    x_outputs_clip = clip_model.inference_by_sim_probs(images1, need_new_text_features=True)
    x_outputs_clip = x_outputs_clip.to(torch.float32)

    x_refine, confidence = refine_x_outputs(x_outputs, x_outputs_clip)
    Clip_KLLoss = KLLoss(x_outputs_clip, x_refine.detach(), confidence, args)
    prompt_loss = coefficient * Clip_KLLoss
    return prompt_loss


def CLDLoss_minN(prob_s, prob_w, mask=None, weights=None, minN=0):
    cl_w, c_w = get_centroids_TopN(prob_w, minN)
    affnity_s2w = torch.mm(prob_s, c_w.t())
    if mask is None:
        loss = F.cross_entropy(affnity_s2w.div(0.07), cl_w, weight=weights)
    else:
        loss = (F.cross_entropy(affnity_s2w.div(0.07), cl_w, reduction='none', weight=weights) * mask).mean()
    return loss


def get_centroids_TopN(prob, minN=0):
    N, D = prob.shape  # N是logits的batch_size D是分类数
    K = D  # K是分类数

    cl = prob.argsort(dim=1, descending=False)
    cl = cl[:, minN]
    cl = cl.long().view(-1)  # -> class index  cl是prob的每个样本中最小的最不可能的序数，然后转换为一维

    Ncl = cl.view(cl.size(0), 1).expand(-1, D)  # 先将cl的shape从N变为N,1，在变成N,K其中元素就cl的扩展，每个N中是全部一样的,赋给Ncl
    unique_labels, labels_count = Ncl.unique(dim=0,
                                             return_counts=True)  # 分别是按照之前的每个样本的最小的那个分类，计算的一个集合和元素分别出现的次数，unique的集合数量，D labels_count的shape集合数量
    labels_count_all = torch.ones([K]).long().cuda()  # -> counts of each class 产生一个长度为K，分类数的tensor，其中元素全是1
    labels_count_all[unique_labels[:, 0]] = labels_count  # 计算每一个类别出现的次数，没出现的记为1
    c = torch.zeros([K, D], dtype=prob.dtype).cuda().scatter_add_(0, Ncl, prob)  # -> class centroids
    c = c / labels_count_all.float().unsqueeze(1)  # 计算出一个按照按照最小的logits聚类后相加的结果，再平均的一个向量
    return cl, c


def mix_CldLoss(x_outputs, mixed_outputs, shuffle_idx1, shuffle_idx2, lam, minN=-1):
    mixed_outputs = torch.softmax(mixed_outputs, dim=1)
    x_outputs = x_outputs.detach()
    cld_Loss1 = CLDLoss_minN(mixed_outputs, x_outputs[shuffle_idx1], mask=lam, minN=minN)
    cld_Loss2 = CLDLoss_minN(mixed_outputs, x_outputs[shuffle_idx2], mask=1 - lam, minN=minN)
    return cld_Loss2 + cld_Loss1


def similarity_MI(x_outputs, sim_matrix):
    sim_matrix = sim_matrix - sim_matrix.min()
    sim_matrix.fill_diagonal_(0)
    x_product = x_outputs.unsqueeze(2) * x_outputs.unsqueeze(1)  # 形状变为 n x m x m

    # 应用关系矩阵作为权重
    weighted_product = x_product * sim_matrix

    # 计算每个向量的最终加权和
    result = weighted_product.sum(dim=(1, 2))  # 对每个向量的加权乘积求和
    result = result / 2
    return result


def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%m-%d___%H-%M__', local_time)
    return str_time


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(args, optimizer, iter_num, max_iter):
    decay = (1 + args.lr_gamma * iter_num / max_iter) ** (-args.lr_power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
        normalize = None

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        # transforms.RandomCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
        normalize = None
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def gmm(all_fea, pi, mu, all_output):  # pi是all_output.sum(dim=0)是各个类别大概的数量先验知识
    Cov = []  # mu是类似于shot的方法一次计算出的特征的聚类中心，
    dist = []  # all_output每个类别平均的概率
    log_probs = []

    for i in range(len(mu)):  # 对每个类进行一次循环
        temp = all_fea - mu[i]  # temp是每个特征去掉类i的中心后的结果
        predi = all_output[:, i].unsqueeze(dim=-1)  # 得到每个输出概率的第i项并在最外层加一个维度变成[50000,1]
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + args.epsilon * torch.eye(
            temp.shape[1]).cuda()
        # 计算一个矩阵，前面是去中心的特征，后续是去中心的特征再通过shot中的聚类的到的中心，求出这两个的相似度
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += args.epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5 * (Covi.shape[0] * np.log(2 * math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)

    return zz, gamma


def evaluation(loader, netF, netB, netC, args, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in tqdm(range(len(loader))):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1].cuda()
        inputs = inputs.cuda()
        feas = netB(netF(inputs))
        outputs = netC(feas)

        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)


    if args.dset == 'VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    if args.dset=='office-home':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)

    top_dis = 1
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon2), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    initc = torch.matmul(aff.t(), (all_fea))
    initc = initc / (1e-8 + aff.sum(dim=0)[:, None])

    if args.pickle and (cnt == 0):
        data = {
            'all_fea': all_fea,
            'all_output': all_output,
            'all_label': all_label,
            'all_fea_orig': all_fea_orig,
        }
        filename = osp.join(args.output_dir, 'data_{}'.format(args.names[args.t]) + args.prefix + '.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('data_{}.pickle finished\n'.format(args.names[args.t]))

    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea), args.class_num) / args.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)

    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)

    aff = gamma

    acc = (pred_label == all_label).float().mean()
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'

    if args.dset == 'VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'
    if args.dset=='office-home':
        log_str += 'office-home classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)

    ############################## Computing JMDS score #############################

    sort_zz, sort_zz_indices = zz.sort(dim=1, descending=True)

    zz_sub = sort_zz[:, 0] - sort_zz[:, 1]

    LPG = zz_sub / zz_sub.max()
    LPG = LPG + sort_zz[:, 0]
    # LPG = zz_sub+sort_zz[:,0]
    # LPG = zz_sub
    indices = torch.arange(all_output.shape[1]).cuda()

    # 使用 unsqueeze() 在第二维添加一个维度
    indices_unsqueezed = indices.unsqueeze(1)

    # 使用 expand() 复制第二维以匹配第一维的大小
    target_tensor = indices_unsqueezed.expand(-1, 2)

    n_cls = all_output.shape[1]
    TOP12_connection = \
        utils.get_class_centroids_TOP2_v2(all_output,  normalize=True)[1]
    TOP12_connection0=TOP12_connection.clone().detach()
    t_f_mat = TOP12_connection.view(-1)
    TOP12_connection.fill_diagonal_(0)
    t_f_mat = TOP12_connection.view(-1)

    top_k = 50#50
    values, indices = t_f_mat.topk(top_k, largest=True)
    indices_x, indices_y = indices // n_cls, indices % n_cls

    sim_indices = torch.stack((indices_x, indices_y), dim=1)
    sim_indices = torch.cat((sim_indices, target_tensor), dim=0)
    selected_sample_index,weight_set = get_selected_sample_index(all_output,sim_indices,all_output.shape[1],TOP12_connection0)


    if args.coeff == 'JMDS':
        PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
    elif args.coeff == 'PPL':
        JMDS = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    elif args.coeff == 'NO':
        JMDS = torch.ones_like(LPG)
    else:
        JMDS = LPG

    sample_weight = JMDS

    if args.dset == 'VISDA-C':
        return sample_weight, aacc / 100, sim_indices,TOP12_connection0,selected_sample_index,all_fea,weight_set
    return  sample_weight, accuracy, sim_indices,TOP12_connection0,selected_sample_index,all_fea,weight_set


def KLLoss(input_, target_, coeff, args):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + args.epsilon2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)


def NewKLLoss(input_, target_, coeff, args):
    log_softmax = F.log_softmax(input_, dim=1)
    # 使用 PyTorch 的 kl_div 计算 KL 散度，target_ 应该是概率分布
    kl_loss = F.kl_div(log_softmax, target_, reduction='none').sum(dim=1)
    # 应用权重和求均值
    kl_loss *= coeff
    return kl_loss.mean(dim=0)


def DKL_decouple(a, isSoftmax=True):
    max_vals, max_indices = torch.max(a, dim=1, keepdim=True)
    batch_size, class_num = a.shape

    # 步骤2: 创建 b
    b = torch.zeros(batch_size, 2)
    b[:, 0] = max_vals.squeeze()  # 最大值
    b[:, 1] = 1 - b[:, 0]  # 其他值的和，由于总和为1，直接用 1 减去最大值即可

    # 步骤3: 创建 c
    c = torch.zeros_like(a)
    c[:, :] = a[:, :]
    c.scatter_(1, max_indices, float('-inf'))  # 将最大值位置设置为负无穷，以便在 softmax 时忽略
    c = torch.nn.functional.softmax(c, dim=1)  # 对修改后的 tensor 应用 softmax

    # 移除负无穷位置的列（即最大值位置的列）
    c = c.masked_select(c != 0).view(batch_size, class_num - 1)
    return b.cuda(), c.cuda()


def DKL(input_, target_, coeff, args):
    target_ = target_.detach()
    select = torch.eq(torch.argmax(input_, dim=1), torch.argmax(target_, dim=1))

    input_DKL = input_[select]
    target_DKL = target_[select]
    coeff_DKL = coeff[select]

    input_KL = input_[~select]
    target_KL = target_[~select]
    coeff_KL = coeff[~select]

    loss_kl = NewKLLoss(input_KL, target_KL, coeff_KL, args)

    input_TCKL, input_NCKL = DKL_decouple(input_DKL)
    target_TCKL, target_NCKL = DKL_decouple(target_DKL)
    coeff_TCKL, coeff_NCKL = coeff_DKL.cuda(), coeff_DKL.cuda()
    loss_tckl = KLLoss(input_TCKL, target_TCKL, coeff_TCKL, args)
    loss_nckl = KLLoss(input_NCKL, target_NCKL, coeff_NCKL, args)

    lam_kl, lam_tckl, lam_nckl = 1, 1, 1

    loss = loss_kl * lam_kl + loss_tckl * lam_tckl + loss_nckl * lam_nckl

    return loss


def mixup(x , netF, netB, netC, outputs_clip_sim,
          attention_net,res_fea_set,clip_fea_set,args):
    # weight mixup
    contrast_loss = utils.ContrastiveLoss1()
    x_fea = netB(netF(x))
    x_outputs = torch.nn.functional.softmax(netC(x_fea), dim=1)
        ###########更改

    x_fea_aug,Fsa,Fta,Fsp,Ftp= attention_net(x_fea,res_fea_set.cuda(),clip_fea_set.cuda())
    y_aug = torch.softmax(netC(x_fea_aug),dim=1)

    outputs_clip = outputs_clip_sim
    outputs_clip = outputs_clip.cuda()
    outputs_clip = outputs_clip.detach()


    Clip_KL_lambda = 0.1
    x_outputs_refine, confidence = utils.refine_x_outputs(x_outputs.detach(), outputs_clip)


    same_loss = NewKLLoss(y_aug,x_outputs_refine,torch.ones_like(confidence),args)
    contrastive_loss = contrast_loss(Fsa,Fta)
    Clip_KLLoss = KLLoss(x_outputs, x_outputs_refine, torch.ones_like(confidence), args)
    Total_Loss =    Clip_KLLoss * Clip_KL_lambda+same_loss*0.1+contrastive_loss*0.1

    return 3*Total_Loss, x_outputs








def train_target(args):
    if args.dset == 'VISDA-C':
        classnames = ['plane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
                      'skateboard', 'train', "truck"]
        ctx_init = 'A picture of a'
    elif args.dset == 'office-home':
        classnames = [
            "alarm clock", "backpack", "batteries", "bed", "bike",
            "bottle", "bucket", "calculator", "calendar", "candles",
            "chair", "clipboards", "computer", "couch", "curtains",
            "desk lamp", "drill", "eraser", "exit sign", "fan",
            "file cabinet", "flipflops", "flowers", "folder", "fork",
            "glasses", "hammer", "helmet", "kettle", "keyboard",
            "knives", "lampshade", "laptop", "marker", "monitor",
            "mop", "mouse", "mug", "notebook", "oven",
            "pan", "paper clip", "pen", "pencil", "post-it notes",
            "printer", "push pin", "radio", "refrigerator", "ruler",
            "scissors", "screwdriver", "shelf", "sink", "sneakers",
            "soda", "speaker", "spoon", "TV", "table", "telephone",
            "toothbrush", "toys", "trash can", "webcam"
        ]
        ctx_init = 'A picture of a'
    elif args.dset == 'DomainNet126':
        classnames = []
        with open("/home/q23301278/Codes/COVA-JMDS/data/DomainNet126/classname.txt", 'r') as file:
            # 使用readlines()方法读取文件的所有行
            lines = file.readlines()

            # 遍历每一行，strip()方法用于移除末尾的换行符
            for line in lines:
                classnames.append(line.strip())
        ctx_init = 'A picture of a'
    else:
        classnames = []

    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F_{}.pt'.format(2020)
    print('modelpath: {}'.format(modelpath))
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B_{}.pt'.format(2020)
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C_{}.pt'.format(2020)
    netC.load_state_dict(torch.load(modelpath))

    attention_net = utils.IFT_Module_fea(256,0.1,0.1)
    attention_net = attention_net.cuda()

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    for k, v in netC.named_parameters():
        if args.lr_decay3 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay3}]
        else:
            v.requires_grad = False
    for k,v in attention_net.named_parameters():
        param_group+= [{'params': v, 'lr': args.lr *args.lr_decay_attention}]
    crop_size = 224
    augment1 = transforms.Compose([
        # transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
    # prompt_optimizer

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    cnt = 0

    dset_loaders = data_load(args)

    epochs = []
    accuracies = []

    netF.eval()
    netB.eval()
    netC.eval()

    clip_model = prompt_tune.get_coop(args.clip_backbone, args.dset, args.gpu_id, 16,
                                      ctx_init, None, logit_rate=0.01,fine_gained_sim=True)
    clip_model = clip_model.cuda()
    for name, param in clip_model.named_parameters():
        if "prompt_learner" not in name or name == 'prompt_learner.clip_model.token_embedding.weight':
            param.requires_grad_(False)
        else:
            print(name)

    if 'RN' in args.clip_backbone:
        prompt_lr = 1e-4
    else:
        prompt_lr = 1e-3
    optimizer_prompt_clip = torch.optim.SGD([clip_model.prompt_learner.ctx], prompt_lr, weight_decay=5e-4, momentum=0.9, nesterov=False)
    optimizer_refine_clip = torch.optim.SGD([clip_model.prompt_learner.sim], prompt_lr, weight_decay=5e-4, momentum=0.9, nesterov=False)

    with torch.no_grad():
        coeff, accuracy, sim_indices,prob_center,res_selected_sample_index,all_fea,weight_set_res = evaluation(
            dset_loaders["test"], netF, netB, netC, args, cnt
        )

        clip_model.sim_matrix = prob_center
        clip_model.modify_sim_text(sim_indices)
        clip_model.reset_classnames_sim_gen_beifen(classnames,args.clip_backbone,prob_center)

    with torch.no_grad():

        all_outputs_clip, all_outputs_clip_sim,clip_selected_sample_index,weight_set_clip = update_clip_outputs(dset_loaders["test"], clip_model)
        res_fea_set = fuse_fea(all_fea,res_selected_sample_index,weight_set_res)
        clip_fea_set = fuse_fea(all_fea,clip_selected_sample_index,weight_set_clip)

    netF.train()
    netB.train()
    netC.train()
    clip_model.train()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // (args.interval)
    iter_num = 0

    print('\nTraining start\n')
    while iter_num < max_iter:
        try:
            inputs_test, label, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, label, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)


        images1 = torch.autograd.Variable(augment1(inputs_test))
        images1 = images1.cuda()



        CoWA_loss, x_outputs = mixup(images1, netF, netB, netC,
                                     all_outputs_clip_sim[tar_idx].detach(),attention_net, res_fea_set,clip_fea_set,args)



        optimizer.zero_grad()  # 计算第一个损失的梯度
        CoWA_loss.backward()  # 需要保留图以计算第二个损失的梯度
        # 清除第二个优化器的梯度
        optimizer.step()

        Prompt_loss = prompt_adjust(images1, x_outputs.detach(), clip_model, iter_num, max_iter, interval_iter, prob_center,args)
        optimizer_prompt_clip.zero_grad()
        Prompt_loss.backward()
        optimizer_prompt_clip.step()


        Refine_loss = Refine_sim_Loss_fine_gain(clip_model,prob_center)*0.1#最好效果目前是
        optimizer_refine_clip.zero_grad()
        Refine_loss.backward()
        optimizer_refine_clip.step()




        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('Evaluation iter:{}/{} start.'.format(iter_num, max_iter))
            log_str = 'Task: {}, Iter:{}/{};'.format(args.name, iter_num, max_iter)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)

            netF.eval()
            netB.eval()
            netC.eval()

            cnt += 1
            with torch.no_grad():
                # Compute JMDS score at offline & evaluation.
                coeff, accuracy, sim_indices,prob_center,res_selected_sample_index,all_fea,weight_set_res = evaluation(dset_loaders["test"],
                                                                                                  netF, netB, netC,
                                                                                                  args, cnt)
                epochs.append(cnt)
                accuracies.append(np.round(accuracy * 100, 2))
                all_outputs_clip, all_outputs_clip_sim,clip_selected_sample_index,weight_set_clip = update_clip_outputs(dset_loaders["test"], clip_model)
                clip_model.modify_sim_text(sim_indices)
                clip_model.reset_classnames_sim_gen_beifen(classnames, args.clip_backbone, prob_center)
                res_fea_set = fuse_fea(all_fea, res_selected_sample_index,weight_set_res)
                clip_fea_set = fuse_fea(all_fea, clip_selected_sample_index,weight_set_clip)

            print('Evaluation iter:{}/{} finished.\n'.format(iter_num, max_iter))

            netF.train()
            netB.train()
            netC.train()

    ####################################################################
    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, 'ckpt_F_' + args.prefix + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, 'ckpt_B_' + args.prefix + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, 'ckpt_C_' + args.prefix + ".pt"))

    log_str = '\nAccuracies history : {}\n'.format(accuracies)
    best_acc = max(accuracies)
    log_best = f"The best Accuracies = {best_acc}\n"
    args.out_file.write(log_str)
    args.out_file.write(log_best)
    args.out_file.flush()
    print(log_str)
    print(log_best)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, accuracies, 'o-')
    plt.savefig(osp.join(args.output_dir, 'png_{}.png'.format(args.prefix)))
    plt.close()

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'DomainNet126'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--warm', type=float, default=0.0)
    parser.add_argument('--coeff', type=str, default='JMDS0', choices=['LPG', 'JMDS', 'PPL', 'NO', 'JMDS0'])
    parser.add_argument('--pickle', default=False, action='store_true')
    parser.add_argument('--lr_gamma', type=float, default=10.0)
    parser.add_argument('--lr_power', type=float, default=0.75)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0.1)
    parser.add_argument('--lr_decay_attention', type=float, default=0.1)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--epsilon2', type=float, default=1e-6)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--n_power', type=int, default=1)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--issave', type=bool, default=False)
    parser.add_argument("--random_mode", type=int, default=1)
    parser.add_argument("--clip_backbone", type=str, default='ViT-B/16')

    args = parser.parse_args()

    if args.dset == 'office-home':
        args.names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        args.names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        args.names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        args.names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DomainNet126':
        args.names = ['clipart','painting', 'real', 'sketch']
        args.class_num = 126

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import utils
    import prompt_tune

    if args.random_mode == 1:
        SEED = random.randint(0, 9999999)
    else:
        SEED = args.seed

    ############# If you want to obtain the stochastic result, comment following lines. #############
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    annotation = f"coeff={args.coeff}   batch_size={args.batch_size}   SEED={SEED}  " + "\n"
    print(annotation)

    init_t = args.t
    for i in range(len(args.names)):
        start = time.time()
        if init_t == args.s:
            init_t = init_t + 1
            continue
        args.t = init_t
        init_t = (init_t + 1) % len(args.names)

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + args.names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + args.names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + args.names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, args.names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset,
                                   args.names[args.s][0].upper() + args.names[args.t][0].upper())
        args.name = args.names[args.s][0].upper() + args.names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.prefix = '{}_alpha{}_lr{}_epoch{}_interval{}_seed{}_warm{}_{}'.format(
            args.coeff, args.alpha, args.lr, args.max_epoch, args.interval, SEED, args.warm, get_current_time()
        )

        ####################################################################
        if not osp.exists(osp.join(args.output_dir, 'ckpt_F_' + args.prefix + ".pt")):
            args.out_file = open(osp.join(args.output_dir, 'log' + args.prefix + '.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            # 写注释到文件中

            args.out_file.write(annotation)

            args.out_file.flush()
            #start train
            train_target(args)

            total_time = time.time() - start
            log_str = 'Consumed time : {} h {} m {}s'.format(total_time // 3600, (total_time // 60) % 60,
                                                             np.round(total_time % 60, 2))
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
        else:
            print('{} Already exists'.format(osp.join(args.output_dir, 'log' + args.prefix + '.txt')))



