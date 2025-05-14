from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import os
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from torch_geometric.loader import DataLoader
from models.SD2 import SD2
from loaddataset_new2 import GraphDataset5
from config import get_argv
from utils_new import GradCAM, GradCAMNode, clourMol, StripAlphaFromImage, TrimImgByWhite
from rdkit import Chem
from rdkit.Chem import Draw
from pathlib import Path
from collections import defaultdict
import os

def weighted_fingerprint(mol, weight):
    # 检查权重数组的长度是否与分子中的原子数量匹配
    num_atoms = mol.GetNumAtoms()
    if len(weight) != num_atoms:
        raise ValueError(f"权重数组的长度 ({len(weight)}) 与分子的原子数量 ({num_atoms}) 不匹配")
    
    # 确保权重数组没有无效值（如 NaN 或无穷大）
    if np.any(np.isnan(weight)) or np.any(np.isinf(weight)):
        raise ValueError("权重数组包含无效值（NaN 或无穷大）")

    # 创建一个绘图对象（大小可根据需要调整）
    d = Draw.MolDraw2DCairo(600, 600)
    d.ClearDrawing()
    # 获取相似性图（注意，这个函数返回一个 NumPy 数组）
    min_value = min(weight.tolist())
    max_value = max(weight.tolist())
    # draw2d, locs, weights, sigmas, contourLines, ps 
    # sm = SimilarityMaps.GetSimilarityMapFromWeights(mol, weight.tolist(), draw2d=d)
    ###(mol, weights, colorMap=None, scale=-1, size=(250, 250), sigma=None,
                                # coordScale=1.5, step=0.01, colors='k', contourLines=10, alpha=0.5,
                                # draw2d=None, **kwargs):
    sm = SimilarityMaps.GetSimilarityMapFromWeights(mol, weight.tolist(), draw2d=d,contourLines=20, scale=2, alpha=1,vmin=min_value, vmax=max_value)
    # Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=ps)

    # 完成绘制
    d.FinishDrawing()

    # 转换绘图结果为图像
    img_data = d.GetDrawingText()
    img = PILImage.open(io.BytesIO(img_data))
    return img


def test_drug_vis(model, data, device):
    
    model.eval()
    
    gradcam = GradCAM(model, target_layers = model.readouts[-1])

    out_path = Path('checkpoint/model1/att_molpng/vis')

    for d1, d2, d1_edge, d2_edge, _, drug_info in data:

        d1 = d1.to(device)
        d2 = d2.to(device)
        d1_edge = d1_edge.to(device)
        d2_edge = d2_edge.to(device)

        name1 = drug_info['drug1-name'][0]
        name2 = drug_info['drug2-name'][0]
        smile1 = drug_info['drug1-smile'][0]
        smile2 = drug_info['drug2-smile'][0]
        input_ = (d1, d2, d1_edge, d2_edge) 

        cam1, cam2 = gradcam.calculate_cam(input_)
        gradcam.clear()
        
        out_dir = out_path / f'{name1}'
        out_dir.mkdir(exist_ok=True)

        mol1 = Chem.MolFromSmiles(smile1)
        mol2 = Chem.MolFromSmiles(smile2)

        # node_index1 = np.where(cam1>0)[0]
        # node_index2 = np.where(cam2>0)[0]
        img1 = weighted_fingerprint(mol1, cam1)
        img2 = weighted_fingerprint(mol2, cam2)

        img1 = StripAlphaFromImage(img1)
        img1 = TrimImgByWhite(img1)
        img1.save(str(out_dir / f'{name1}_{name2}.png'),dpi=(600, 600))

        img2 = StripAlphaFromImage(img2)
        img2 = TrimImgByWhite(img2)
        img2.save(str(out_dir / f'{name2}_{name1}.png'),dpi=(600, 600))

        cam1_ls = cam1.tolist()
        cam2_ls = cam2.tolist()

        for i, at in enumerate(mol1.GetAtoms()):
            lbl = '%s:%.2f'%(at.GetSymbol(), cam1_ls[i])
            at.SetProp('atomLabel', lbl)

        for i, at in enumerate(mol2.GetAtoms()):
            lbl = '%s:%.2f'%(at.GetSymbol(), cam2_ls[i])
            at.SetProp('atomLabel', lbl)
        
        Draw.MolToFile(mol1, str(out_dir / f'{name1}_{name2}_dis-att.png'), size=(600, 600))
        Draw.MolToFile(mol2, str(out_dir / f'{name2}_{name1}_dis-att.png'), size=(600, 600))

if __name__ == '__main__':
    args = get_argv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()

    # 定义数据集
    train_dataset = GraphDataset5(args.train_data, args=args, bidrection=True,norm_size=True)
    val_dataset = GraphDataset5(args.val_data, args=args, bidrection=True, norm_size=train_dataset.norm_size)
    test_dataset = GraphDataset5(args.test_data, args=args, bidrection=True,norm_size=train_dataset.norm_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)    

    model = SD2(args.nclass, args.natom, in_edge_features=args.n_edge_attr, hidd_dim=args.hidden,  
                    n_out_feats=args.n_out_feats, n_heads=args.nb_heads, edge_feature=64, dp=args.dropout).to(device)
    
    # 加载参数
    ckp = torch.load(args.load_checkpoint)
    model.load_state_dict(ckp)
    test_drug_vis(model, test_loader, device) # 分子级注意力可视化

