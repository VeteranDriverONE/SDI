import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import random

from torch import optim, nn
from torch_geometric.loader import DataLoader
from config import get_argv
from training_func import training_classification, evaluate_classification
from models.GAT import GAT, SpGAT, GAT2, SpGAT2, AttentiveFP2
from models.SD2 import SD2
from models.loss import focal_loss, binary_cross_entropy
from torch.utils.tensorboard import SummaryWriter
from loaddataset_new2 import GraphDataset5
from sklearn.metrics import (roc_curve, auc, precision_score, recall_score, classification_report, precision_recall_curve,accuracy_score, roc_auc_score, f1_score,
                                mean_squared_error, mean_absolute_error, r2_score)
from pathlib import Path

if __name__ == '__main__':
    args = get_argv()

    if args.device>=0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    args.use_adj = False
    writer = SummaryWriter('tb')

    # 数据集加载, 邻接
    if args.use_adj:
        args.batch_size = 1
        
    train_dataset = GraphDataset5(args.train_data, args, bidrection=True,norm_size=True)
    val_dataset = GraphDataset5(args.val_data, args, bidrection=True, norm_size=train_dataset.norm_size)
    test_dataset = GraphDataset5(args.test_data, args, bidrection=True,norm_size=train_dataset.norm_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型加载
    if args.use_adj:
        model = GAT2(args.natom,args.hidden, args.nclass, args.dropout, args.alpha, args.nb_heads)
    else:
        model = SD2(args.nclass, args.natom, in_edge_features=args.n_edge_attr, hidd_dim=args.hidden,  
                    n_out_feats=args.n_out_feats, n_heads=args.nb_heads, edge_feature=64, dp=args.dropout).to(device)

    if args.load_checkpoint is not None:
        ckp = torch.load(args.load_checkpoint)
        model.load_state_dict(ckp)
        print('Pre-trained model loaded')
    # 定义优化器损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    # criterion = focal_loss(alpha=[1,0.5], gamma=2, num_classes=2, size_average=True)
    criterion = binary_cross_entropy

    # 训练
    # min_val_loss = torch.inf
    best_Performance = 0
    best_t = 0
    total_loss_ls = []
    val_loss_ls = []
    for epoch in range(1, args.epochs+1):
        print('Train Epoch:{}, Start time:{}'.format( epoch, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) )) 
        total_loss = training_classification(model, train_loader, optimizer, criterion, device)      
        writer.add_scalar(f'train/loss', total_loss, epoch)
        total_loss_ls.append(total_loss)
        print(f'Train Epoch:{epoch}, train loss:{total_loss}')

        if epoch % args.val_gap == 0:
            print('Validate Epoch:{}, Start time:{}'.format( epoch, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) ))
            val_loss, acc, auc_roc, auc_prc, f1, precision, recall = evaluate_classification(model, val_loader, nn.BCELoss(), device)
            writer.add_scalar(f'val/loss', val_loss, epoch)            
            val_loss_ls.append(val_loss)
            print(f'Validate Epoch:{epoch}, val loss:{val_loss}')
            writer.add_scalar(f'val/acc', acc, epoch)
            writer.add_scalar(f'val/auc_roc', auc_roc, epoch)
            writer.add_scalar(f'val/auc_prc', auc_prc, epoch)
            writer.add_scalar(f'val/precision', precision, epoch)
            writer.add_scalar(f'val/auc_recall', recall, epoch)
            writer.add_scalar(f'val/f1', f1, epoch)
            if f1 > best_Performance: # 验证精度大于当前最佳精度，最佳精度就保存
                best_t = epoch
                best_Performance = f1
                torch.save(model.state_dict(), args.save_dir + f'/best_model_{round(best_Performance, 4)}.pt')
                torch.save(model.state_dict(), args.save_dir + f'/best_model.pt')

    # print(f'Best Performance Epoch: {best_t}')
    # print(f'\t\tval_loss:{val_loss:.4f}, val_acc: {acc:.4f}, val_f1: {f1:.4f}')
    # print(f'\t\tval_pre: {precision:.4f},val_re: {recall:.4f}')

    plt.figure(figsize=(6,5)) # 图片宽高
    ax = plt.gca()
    plt.setp(ax.spines.values(), linewidth = 2, edgecolor='black') # 边框宽度为2， 颜色为黑
    plt.plot(range(1, len(total_loss_ls)+1), total_loss_ls, linewidth=2)
    plt.plot(range(args.val_gap, epoch+1, args.val_gap), val_loss_ls, linewidth=2, linestyle='--')
    plt.rcParams['font.sans-serif']=['Arial'] # 字体
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Epoch', fontsize=28)
    plt.ylabel('Convergence', fontsize=28)
    plt.legend(['Train loss','Validate loss'], fontsize=20, frameon=False) # frameon去掉图例边界框
    # plt.title('Convergence Curve', fontsize=20, fontweight='bold')
    plt.gcf().subplots_adjust(left=0.19, bottom=0.19)
    pic_save_path = Path(args.save_dir) / 'att_molpng'
    if not pic_save_path.exists():
        pic_save_path.mkdir(exist_ok=True)
    plt.savefig(pic_save_path / 'train_loss.png')
    plt.close()
    
    print('test:')
    # ckp = torch.load(args.load_checkpoint)
    ckp = torch.load(Path(args.save_dir) / 'best_model.pt')
    model.load_state_dict(ckp)
    test_loss, acc, auc_roc, auc_prc, f1, precision, recall = evaluate_classification(model, test_loader, nn.BCELoss(), device)
    # test_loss, acc, auc_prc, f1, precision, recall = evaluate_classion(model, test_loader_f, criterion, device)
    print(f'epoch:{best_t}, test loss:{test_loss}')
    print(f'\t\ttest_loss:{test_loss:.4f},test_acc: {acc:.4f},test_auc_roc:{auc_roc:.4f},test_f1: {f1:.4f}')
    print(f'\t\ttest_pre: {precision:.4f},test_re: {recall:.4f},test_auc_prc:{auc_prc:.4f}')