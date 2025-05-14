# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F

def BCE_loss(y_hat, y):
	# y_hat:预测标签，已经过sigmoid/softmax处理 shape is (batch_size, 1)
	# y：真实标签（一般为0或1） shape is (batch_size)
    y_hat = torch.cat((1-y_hat, y_hat), 1) # 将二种情况的概率都列出，y_hat形状变为(batch_size, 2)
    # 按照y标定的真实标签，取出预测的概率，来计算损失
    return - torch.log(y_hat.gather(1, y.view(-1, 1))).mean()

def binary_cross_entropy(predict, target, weight=None, size_average=True):
    if not target.detach().cpu().numpy().any() and not predict.detach().cpu().numpy().any():
        return torch.tensor(0)
    # 计算负对数似然
    # loss = -weight[0] * (target * torch.log(predict + 1e-10) + (1 - target) * torch.log(1 - predict + 1e-10))
    loss = -weight * (target * torch.log(predict + 1e-10) + (1 - target) * torch.log(1 - predict + 1e-10))
    # 如果size_average为True，则对损失进行平均
    if size_average:
        return loss.mean()
    else:
        return loss.sum()

class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, torch.Tensor):
            assert len(alpha) == num_classes
            self.alpha = alpha
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
        preds_softmax = preds_softmax.gather(1, labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss