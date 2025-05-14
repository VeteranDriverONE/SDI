import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

def csv2tsv(csv_path, tsv_path):
    data = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    data.to_csv(tsv_path, index=False, sep='\t', encoding='utf-8')

def tsv2csv(tsv_path, csv_path):
    data = pd.read_csv(tsv_path, sep='\t')
    data.to_csv(csv_path, index=False)

# 随机划分数据集
def random_split(source_path, out_path, test_part, val_part=None):
    data = pd.read_csv(source_path, sep='\t')
    if val_part is None:
        # 分为训练和测试集
        train_df, test_df = train_test_split(data, test_size=test_part, random_state=1)
        train_df.to_csv(out_path + '\\traindata.tsv', index=False, sep='\t', encoding='utf-8')
        test_df.to_csv(out_path + '\\testdata.tsv', index=False, sep='\t', encoding='utf-8')
    else:
        # 分为训练、验证和测试集
        assert test_part + val_part <1, '测试集+验证集比例下小于1'
        num_sample = len(data)
        train_df, test_df = train_test_split(data, test_size=test_part, random_state=1)
        new_val = num_sample*val_part / len(train_df)
        train_df, val_df = train_test_split(train_df, test_size=new_val, random_state=1)
        train_df.to_csv(out_path + '\\traindata.tsv', index=False, sep='\t', encoding='utf-8')
        test_df.to_csv(out_path + '\\testdata.tsv', index=False, sep='\t', encoding='utf-8')
        val_df.to_csv(out_path + '\\valdata.tsv', index=False, sep='\t', encoding='utf-8')

# 指定药物划分数据集
def split_dataset2(source_path, out_path, son_names=[], drug_names=[]):
    data = pd.read_csv(source_path, sep=',')
    test_data = data[data['sonosensitizer'].isin(son_names) | data['drug'].isin(drug_names)]
    train_data = data.drop(data[data['sonosensitizer'].isin(son_names) | data['drug'].isin(drug_names)].index)
    test_data = test_data.drop(test_data[test_data['sonosensitizer'].isin(son_names) & test_data['drug'].isin(drug_names)].index)
    train_data.to_csv(out_path + '\\traindata25.csv', index=False, sep=',', encoding='utf-8')
    test_data.to_csv(out_path + '\\testdata25.csv', index=False, sep=',', encoding='utf-8')

# 指定药物划分数据集
def split_dataset3(source_path, out_path, son_names=[], drug_names=[]):
    data = pd.read_csv(source_path, sep=',')
    test_data = data[data['sonosensitizer'].isin(son_names)]
    new_data = data.drop(data[data['sonosensitizer'].isin(son_names)].index)
    test_data = test_data[test_data['drug'].isin(drug_names)]
    new_data = new_data.drop(new_data[new_data['drug'].isin(drug_names)].index)
    new_data.to_csv(out_path + '\\traindata.csv', index=False, sep=',', encoding='utf-8')
    test_data.to_csv(out_path + '\\testdata.csv', index=False, sep=',', encoding='utf-8')
 
# 药物名改id
def drugname2drugid(tsv_path, out_path,  key='drug-en'):
    data = pd.read_csv(tsv_path, sep='\t')
    map_file = pd.read_csv('D:\workspace\SDI-Public\dataset\\name-dict.csv', sep=',', encoding='utf-8')
    data_dict = {}
    for i in range(len(map_file)):
        data_dict[map_file[key][i]] = map_file['drug-id'][i]
    data_copy = data.copy()
    for i in range(len(data)-1,-1,-1):
        son = data_copy['sonosensitizer'][i]
        drug = data_copy['drug'][i]
        son = data_dict.get(son, 0)
        drug = data_dict.get(drug, 0)
        if son != 0 and drug != 0:
            data_copy.loc[i, 'sonosensitizer'] = son
            data_copy.loc[i, 'drug'] = drug
        else:
            data_copy = data_copy.drop(i)
    data_copy.to_csv(out_path, index=False, sep='\t', encoding='utf-8')

# 绘图
def _drawerToImage(d2d):
    try:
        import Image
    except ImportError:
        from PIL import Image
    sio = BytesIO(d2d.GetDrawingText())
    return Image.open(sio)

def get_atom_color(atom):
    """Get the color for a given atom based on its element."""
#,,,,,'Si','P',,'Br','Mg',,'Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn'
#,'Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
    color_map = {
        'C': (0.7, 0.13, 0.13, 0.5),  
        'O': (1.0, 0.0, 0.0, 0.5),  
        'N': (0.0, 0.0, 1, 0.5),  
        'F': (0.0, 1.0, 1.0, 0.5),  
        'Cl': (0.0, 1.0, 0.0, 0.5),  
        'S': (1.0, 0.5, 0.0, 0.5),
    }
    return color_map.get(atom, (0.5, 0.5, 0.5,0.5))  # 默认灰色

def clourMol(mol,highlightAtoms_p=None,highlightAtomColors_p=None,highlightBonds_p=None,highlightBondColors_p=None,sz=[600,600]):
    d2d = rdMolDraw2D.MolDraw2DCairo(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 20
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    if highlightAtoms_p:
        atom_types = [mol.GetAtomWithIdx(int(idx)).GetSymbol() for idx in highlightAtoms_p]
        # Generate colors for each highlighted atom based on its type
        # highlightAtomColors_p = [get_atom_color(atom) for atom in atom_types]
        highlightAtomColors_p = {idx: get_atom_color(atom) for idx, atom in zip(highlightAtoms_p, atom_types)}
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p, highlightAtomColors=highlightAtomColors_p, highlightBonds=highlightBonds_p, highlightBondColors=highlightBondColors_p)
    d2d.FinishDrawing()
    product_img=_drawerToImage(d2d)
    return product_img

def StripAlphaFromImage(img):
    '''This function takes an RGBA PIL image and returns an RGB image'''
    if len(img.split()) == 3:
        return img
    return Image.merge('RGB', img.split()[:3])

def TrimImgByWhite(img, padding=10):
    '''This function takes a PIL image, img, and crops it to the minimum rectangle
    based on its whiteness/transparency. 5 pixel padding used automatically.'''
    # Convert to array
    as_array = np.array(img)  # N x N x (r,g,b,a)
    # Set previously-transparent pixels to white
    if as_array.shape[2] == 4:
        as_array[as_array[:, :, 3] == 0] = [255, 255, 255, 0]
    as_array = as_array[:, :, :3]
    # Content defined as non-white and non-transparent pixel
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)
    # Crop down
    margin = 5
    x_range = max([min(xs) - margin, 0]), min([max(xs) + margin, as_array.shape[0]])
    y_range = max([min(ys) - margin, 0]), min([max(ys) + margin, as_array.shape[1]])
    as_array_cropped = as_array[
        x_range[0]:x_range[1], y_range[0]:y_range[1], 0:3]
    img = Image.fromarray(as_array_cropped, mode='RGB')
    return ImageOps.expand(img, border=padding, fill=(255, 255, 255))

class GradCAM():
    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        self.activations = []
        self.grads = []
        
    def forward_hook(self, module, input, output):
        self.activations.append(output[0])
        
    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())
        
    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)                 # Module.to() is in-place method 
            d1, d2, d1_edge, d2_edge = model_input    
            d1 = d1.to(device)
            d2 = d2.to(device)
            d1_edge = d1_edge.to(device)
            d2_edge = d2_edge.to(device)    
        self.model.eval()
        # forward
        y_hat = self.model(d1, d2, d1_edge, d2_edge)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)
        # backward
        self.model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()
        # get activations and gradients
        # 注：先前向的，梯度后到达；前向和梯度顺序相反
        mol1_activations = self.activations[0].permute(1,0).cpu().data.numpy().squeeze()
        mol1_grads = self.grads[1].permute(1,0).cpu().data.numpy().squeeze()
        mol2_activations = self.activations[1].permute(1,0).cpu().data.numpy().squeeze()
        mol2_grads = self.grads[0].permute(1,0).cpu().data.numpy().squeeze()
        
        # calculate weights
        weights = np.mean(mol1_grads.reshape(mol1_grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1) # D, N
        cam1 = (weights * mol1_activations).sum(axis=0)
        cam1 = np.maximum(cam1, 0) # ReLU
        cam1 = cam1 / cam1.max()

        weights = np.mean(mol2_grads.reshape(mol2_grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1) # D, N
        cam2 = (weights * mol2_activations).sum(axis=0)
        cam2 = np.maximum(cam2, 0) # ReLU
        cam2 = cam2 / cam2.max()

        return cam1, cam2
    
    def clear(self):
        
        self.activations = []
        self.grads = []


class GradCAM():
    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        self.activations = []
        self.grads = []
        
    def forward_hook(self, module, input, output):
        self.activations.append(output[0])
        
    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())
        
    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)                 # Module.to() is in-place method 
            d1, d2, d1_edge, d2_edge = model_input    
            d1 = d1.to(device)
            d2 = d2.to(device)
            d1_edge = d1_edge.to(device)
            d2_edge = d2_edge.to(device)    
        self.model.eval()
        # forward
        y_hat = self.model(d1, d2, d1_edge, d2_edge)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)
        # backward
        self.model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()
        # get activations and gradients
        # 注：先前向的，梯度后到达；前向和梯度顺序相反
        mol1_activations = self.activations[0].permute(1,0).cpu().data.numpy().squeeze()
        mol1_grads = self.grads[1].permute(1,0).cpu().data.numpy().squeeze()
        mol2_activations = self.activations[1].permute(1,0).cpu().data.numpy().squeeze()
        mol2_grads = self.grads[0].permute(1,0).cpu().data.numpy().squeeze()
        
        # calculate weights
        weights = np.mean(mol1_grads.reshape(mol1_grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1) # D, N
        cam1 = (weights * mol1_activations).sum(axis=0) # sum(D,1 * D,N)，对每个通道加权，加权后，按节点求和
        cam1 = np.maximum(cam1, 0) # ReLU（绘制注意力图时为了有颜色变化将ReLU去掉）
        cam1 = cam1 / cam1.max()
        # cam1 = 2 * (cam1 - cam1.min()) / (cam1.max() - cam1.min()) - 1 #将cam1中的数值映射到【-1，1】之间
        weights = np.mean(mol2_grads.reshape(mol2_grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1) # D, N
        cam2 = (weights * mol2_activations).sum(axis=0)
        cam2 = np.maximum(cam2, 0) # ReLU （绘制注意力图时为了有颜色变化将ReLU去掉）
        cam2 = cam2 / cam2.max()
        # cam2 = 2 * (cam2 - cam2.min()) / (cam2.max() - cam2.min()) - 1 #将cam2中的数值映射到【-1，1】之间
        return cam1, cam2
    
    def clear(self):
        self.activations = []
        self.grads = []


class GradCAMNode():
    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAMNode).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        self.activations = []
        self.grads = []
    
    def forward_hook(self, module, input, output):
        self.activations.append(output)
        
    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())

    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)                 # Module.to() is in-place method 
            d1, d2, d1_edge, d2_edge = model_input    
            d1 = d1.to(device)
            d2 = d2.to(device)
            d1_edge = d1_edge.to(device)
            d2_edge = d2_edge.to(device)    
        self.model.eval()
        # forward
        y_hat = self.model(d1, d2, d1_edge, d2_edge)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)
        # backward
        self.model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()
        # get activations and gradients
        # 注：先前向的，梯度后到达；前向和梯度顺序相反
        mol1_activations = self.activations[0].permute(1,0).cpu().data.numpy().squeeze()  # D,N
        mol1_grads = self.grads[-1].permute(1,0).cpu().data.numpy().squeeze()
        mol2_activations = self.activations[1].permute(1,0).cpu().data.numpy().squeeze()
        mol2_grads = self.grads[-2].permute(1,0).cpu().data.numpy().squeeze()
        # calculate weights
        weights = np.zeros((11,1))
        out1 = np.zeros((11, mol1_activations.shape[-1]))
        weights[0] = mol1_grads[:43].mean() # 原子符号
        weights[1] = mol1_grads[43:49].mean() # 原子形成键的组数
        weights[2] = mol1_grads[49:57].mean() # 原子电荷
        weights[3] = mol1_grads[57:64].mean() # 杂化
        weights[4] = mol1_grads[64:65].mean() # 是否在环上
        weights[5] = mol1_grads[65:66].mean() # 是否芳香
        weights[6] = mol1_grads[66:67].mean() # 相对原子质量
        weights[7] = mol1_grads[67:68].mean() # 范德华半径
        weights[8] = mol1_grads[68:69].mean() # 共价键半径
        weights[9] = mol1_grads[69:73].mean() # 原子手性
        weights[10] = mol1_grads[73:79].mean() # 
        
        # out[0,:] = mol1_activations[:43].mean(0)
        # out[1,:] = mol1_activations[43:49].mean(0)
        # out[2,:] = mol1_activations[49:57].mean(0)
        # out[3,:] = mol1_activations[57:64].mean(0)
        # out[4,:] = mol1_activations[64:65].mean(0)
        # out[5,:] = mol1_activations[65:66].mean(0)
        # out[6,:] = mol1_activations[66:67].mean(0)
        # out[7,:] = mol1_activations[67:68].mean(0)
        # out[8,:] = mol1_activations[68:69].mean(0)
        # out[9,:] = mol1_activations[69:73].mean(0)
        # out[10,:] = mol1_activations[73:79].mean(0)

        # cam1 = weights * out
        cam1 = mol1_activations*mol1_grads
        # weights = np.mean(mol1_grads.reshape(mol1_grads.shape[0], -1), axis=1)  # D,1
        # weights = weights.reshape(-1, 1)  # D,1
        # cam1 = (weights * mol1_activations).sum(axis=0)
        cam1 = np.maximum(cam1, 0) # ReLU，小于0的调整为0
        # cam1 = cam1 / cam1.max()  # 归一化
        out1[0,:] = cam1[:43].mean(0)
        out1[1,:] = cam1[43:49].mean(0)
        out1[2,:] = cam1[49:57].mean(0)
        out1[3,:] = cam1[57:64].mean(0)
        out1[4,:] = cam1[64:65].mean(0)
        out1[5,:] = cam1[65:66].mean(0)
        out1[6,:] = cam1[66:67].mean(0)
        out1[7,:] = cam1[67:68].mean(0)
        out1[8,:] = cam1[68:69].mean(0)
        out1[9,:] = cam1[69:73].mean(0)
        out1[10,:] = cam1[73:79].mean(0)
        out1 = out1 / out1.max()

        weights = np.zeros((11,1))
        out2 = np.zeros((11, mol2_activations.shape[-1]))

        # weights[0] = mol2_grads[:43].mean() # 原子符号
        # weights[1] = mol2_grads[43:49].mean() # 原子形成键的组数
        # weights[2] = mol2_grads[49:57].mean() # 原子电荷
        # weights[3] = mol2_grads[57:64].mean() # 杂化
        # weights[4] = mol2_grads[64:65].mean() # 是否在环上
        # weights[5] = mol2_grads[65:66].mean() # 是否芳香
        # weights[6] = mol2_grads[66:67].mean() # 相对原子质量
        # weights[7] = mol2_grads[67:68].mean() # 范德华半径
        # weights[8] = mol2_grads[68:69].mean() # 共价键半径
        # weights[9] = mol2_grads[69:73].mean() # 原子手性
        # weights[10] = mol2_grads[73:79].mean() # 
        
        # out[0,:] = mol2_activations[:43].mean(0)
        # out[1,:] = mol2_activations[43:49].mean(0)
        # out[2,:] = mol2_activations[49:57].mean(0)
        # out[3,:] = mol2_activations[57:64].mean(0)
        # out[4,:] = mol2_activations[64:65].mean(0)
        # out[5,:] = mol2_activations[65:66].mean(0)
        # out[6,:] = mol2_activations[66:67].mean(0)
        # out[7,:] = mol2_activations[67:68].mean(0)
        # out[8,:] = mol2_activations[68:69].mean(0)
        # out[9,:] = mol2_activations[69:73].mean(0)
        # out[10,:] = mol2_activations[73:79].mean(0)
        
        # cam2 = weights * out
        cam2 = mol2_activations*mol2_grads
        # weights = np.mean(mol2_grads.reshape(mol2_grads.shape[0], -1), axis=1)
        # weights = weights.reshape(-1, 1) # D, N
        # cam2 = (weights * mol2_activations).sum(axis=0)
        cam2 = np.maximum(cam2, 0) # ReLU
        # cam2 = cam2 / cam2.max()
        out2[0,:] = cam2[:43].mean(0)
        out2[1,:] = cam2[43:49].mean(0)
        out2[2,:] = cam2[49:57].mean(0)
        out2[3,:] = cam2[57:64].mean(0)
        out2[4,:] = cam2[64:65].mean(0)
        out2[5,:] = cam2[65:66].mean(0)
        out2[6,:] = cam2[66:67].mean(0)
        out2[7,:] = cam2[67:68].mean(0)
        out2[8,:] = cam2[68:69].mean(0)
        out2[9,:] = cam2[69:73].mean(0)
        out2[10,:] = cam2[73:79].mean(0)
        out2 = out2 / out2.max()
        # return cam1, cam2
        return out1, out2
    
    def clear(self):
        self.activations = []
        self.grads = []


if __name__ == '__main__':
    # csv2tsv('E:\workspace\Python\SDI\dataset\\all_dataset.csv','E:\workspace\Python\SDI\dataset\\all_dataset.tsv')
    # tsv2csv('E:\workspace\Python\SDI\dataset\\all_dataset.tsv','E:\workspace\Python\SDI\dataset\\all_dataset.csv')
     # random_split('E:\workspace\Python\SDI\dataset\\all_dataset.tsv', out_path='E:\workspace\Python\SDI\dataset\\task3', test_part=0.3)
    # split_dataset2('D:\workspace\SDI-Public\dataset\drugbank\\task3\\new3\\all_dataset_2.csv', out_path='D:\workspace\SDI-Public\dataset\drugbank\\task3\\new3\\5fold_task2_new',
    #               son_names=['Sulfadiazine','Chlorin e6','Sinoporphyrin sodium'], drug_names=['Sunitinib','Imatinib','Mercaptopurine','Doxifluridine'])
    split_dataset3('D:\workspace\SDI-Public\dataset\drugbank\\task3\\new3\\all_dataset_2.csv', out_path='D:\workspace\SDI-Public\dataset\drugbank\\task3\\new3',
                  son_names=['Emodin','Chlorin e6','Hematoporphyrin monomethyl ether','Protoporphyrin IX','Erythrosin B','Indocyanine green','Levofloxacin','Sulfadiazine'], 
                  drug_names=['Gefitinib','Fluorouracil','10-Hydroxycamptothecin','Cyclophosphamide','Methotrexate','Tegafur','Carmustine'])
    # drugname2drugid('D:\workspace\SDI-Public\dataset\drugbank\\task3\\S1_train.csv', 'D:\workspace\SDI-Public\dataset\drugbank\\task3\\S1_train1.csv')
    # drugname2drugid('D:\workspace\SDI-Public\dataset\drugbank\\task3\\S1_test.csv', 'D:\workspace\SDI-Public\dataset\drugbank\\task3\\S1_test1.csv')

