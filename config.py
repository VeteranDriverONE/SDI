import argparse

def get_argv():
    parser = argparse.ArgumentParser()
    # root_path
    parser.add_argument('--save_dir', type=str, default="checkpoint/model6", help='Save dictionary of checkpoint')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path of parameters')
    parser.add_argument('--train_data', type=str, default="", help='Training data set')
    parser.add_argument('--val_data', type=str, default="", help='Validate data set')
    parser.add_argument('--test_data', type=str, default="", help='Test data set')
    parser.add_argument('--son_dict', type=str, default="datasets/selected_excipients_smiles-sample.csv", help='Dictionary of sonosensitizers')
    parser.add_argument('--drug_dict', type=str, default="datasets/selected_drugs_smiles-sample.csv", help='Dictionary of Drug')
    
    # model parameters
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.') # 64/128
    parser.add_argument('--nb_heads', type=list, default=[2,2,2,2,2,2,2,2], help='Number of head attentions.')
    parser.add_argument('--n_out_feats', type=list, default=[64,64,64,64,128,128,256,256], help='Number of head dims.')
    parser.add_argument('--e_out_feats', type=list, default=[64,64,64,64,128,128,256,256],help='Number of edge dims')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--natom', type=int, default=80) # default:58,80含粒径
    parser.add_argument('--n_edge_attr', type=int, default=10) # default:58
    parser.add_argument('--nclass', type=int, default=2)
    # parser.add_argument('--type', type=str, default="classification")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).') #5e-4
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    # train parameters
    parser.add_argument('--device', type=int, default=0, help='The serial number of the GPU used, with -1 indicating CPU used.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.') #800
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')#1e-3（3e-4）
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_gap', type=int, default=5)

    args = parser.parse_args()

    return args