import numpy as np
import pandas as pd
import torch
import json
import warnings
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
from training_func import infer

def test_drug_vis(model, data, device, args):
    model.eval()
    gradcam = GradCAM(model, target_layers = model.readouts[-1])
    # out_path = Path('att_molpng/ClourMol/task3_fold335_drug_vis')
    out_path = Path(args.save_dir) / 'att_molpng' / 'ClourMol'

    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)

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

        node_index1 = np.where(cam1>0)[0]
        node_index2 = np.where(cam2>0)[0]

        img1 = clourMol(mol1, highlightAtoms_p=node_index1.tolist())
        img1 = StripAlphaFromImage(img1)
        img1 = TrimImgByWhite(img1)
        img1.save(str(out_dir / f'{name1}_{name2}.png'),dpi=(600, 600))

        img2 = clourMol(mol2, highlightAtoms_p=node_index2.tolist())
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
    if args.device>=0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    # 定义数据集
    train_dataset = GraphDataset5(args.train_data, args=args, bidrection=True, norm_size=True)
    test_dataset = GraphDataset5(args.test_data, args=args, bidrection=True, norm_size=train_dataset.norm_size)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义模型        
    model = SD2(args.nclass, args.natom, in_edge_features=args.n_edge_attr, hidd_dim=args.hidden, n_out_feats=args.n_out_feats, n_heads=args.nb_heads, 
                edge_feature=64, dp=args.dropout).to(device)
    
    print('test:')
    
    # 加载参数
    if args.load_checkpoint:
        ckp = torch.load(args.load_checkpoint)
        model.load_state_dict(ckp)
        print('参数加载完毕')
    else:
        warnings.warn('未加载任何参数')
    
    infer(model, test_loader, device)
    # test_drug_vis(model, test_loader, device, args)
    