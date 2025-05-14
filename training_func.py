import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (roc_curve, auc, precision_score, recall_score, classification_report, precision_recall_curve,accuracy_score, roc_auc_score, f1_score,
                                mean_squared_error, mean_absolute_error, r2_score)
from models.loss import BCE_loss, binary_cross_entropy

def training_classification(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = []
    for d1, d2, d1_edge, d2_edge, weight, data_dict in dataloader:
        optimizer.zero_grad()
        d1 = d1.to(device)
        d2 = d2.to(device)
        d1_edge = d1_edge.to(device)
        d2_edge = d2_edge.to(device)
        weight = weight.to(device)
        output = model(d1, d2, d1_edge, d2_edge)
        y = d1.y.reshape(-1, output.shape[1]).to(device).float()
        # loss = criterion(output, y)
        loss = criterion(output, y, weight.unsqueeze(-1), True)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    return (sum(total_loss) / len(total_loss))

def evaluate_classification(model, dataloader, criterion, device):
    model.eval()
    total_loss = []
    y_predict = []
    y_pred = []
    y_test = []
    drug1_ = []
    drug2_ = []
    outputs = []
    with torch.no_grad():
        for d1, d2, d1_edge, d2_edge, _, info in dataloader:
            d1 = d1.to(device)
            d2 = d2.to(device)
            d1_edge = d1_edge.to(device)
            d2_edge = d2_edge.to(device)
            output = model(d1, d2, d1_edge, d2_edge)
            drug1_ += info['drug1-name']
            drug2_ += info['drug2-name']
            y = d1.y.reshape(-1, output.shape[1]).to(device).float()            
            total_loss.append(criterion(output,  y).item())
            output = output.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            for i in output:
                y_predict.append(i.argmax())
                y_pred.append(i[1])
            for j in y:
                y_test.append(j.argmax())
            outputs.append(output)
    outputs = np.concatenate(outputs)
    out_pred = np.stack([drug1_, drug2_, y_predict, outputs.max(axis=1), y_test]).transpose()
    print(out_pred)
    out_pred_pd = pd.DataFrame(out_pred)
    out_pred_pd.columns=['Drug1','Drug2','p label','p_value','True label']
    out_pred_pd = out_pred_pd.sort_values(by=['Drug1','Drug2'],)
    # out_pred_pd.to_csv('dataset/validate_pair/result_fold5.csv',index=False)
    acc, auc_roc, auc_prc, f1_score, precision, recall = do_compute_metrics(np.array(y_pred), np.array(y_test))
    PRF = classification_report(y_test, y_predict, target_names=['非纳米','纳米'])
    print(PRF)
    model.train()
    return sum(total_loss) / len(total_loss), acc, auc_roc, auc_prc, f1_score, precision, recall

def infer(model, dataloader, device):
    model.eval()
    y_predict = []
    y_pred = []
    y_test = []
    outputs = []
    drug1_ = []
    drug2_ = []
    with torch.no_grad():
        for d1, d2, d1_edge, d2_edge, _, info in dataloader:
            d1 = d1.to(device)
            d2 = d2.to(device)
            d1_edge = d1_edge.to(device)
            d2_edge = d2_edge.to(device)
            output = model(d1, d2, d1_edge, d2_edge)
            drug1_.append(info['drug1-name'][0])
            drug2_.append(info['drug2-name'][0])
            y = d1.y.reshape(-1, output.shape[1]).to(device).float()            
            output = output.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            for i in output:
                y_predict.append(i.argmax())
                y_pred.append(i[1])
            for j in y:
                y_test.append(j.argmax())
            outputs.append(output)
    acc, auc_roc, auc_prc, f1_score, precision, recall = do_compute_metrics(np.array(y_pred), np.array(y_test))
    print(f'ACC:{acc},F1:{f1_score}, Precision:{precision}, Recall:{recall}')

    outputs = np.concatenate(outputs)
    out_pred = np.stack([drug1_,drug2_,y_predict,outputs.max(axis=1),y_test]).transpose()
    out_pred_pd = pd.DataFrame(out_pred)
    out_pred_pd.columns=['Drug1','Drug2','p label','p_value','True label']
    out_pred_pd = out_pred_pd.sort_values(by=['Drug1','Drug2'],)
    
    return out_pred_pd

def training_regression(model, data, optimizer, criterion, device):
    model.train()
    total_loss = []
    for d1, d2, d1_edge, d2_edge, weight, _ in data:
        optimizer.zero_grad()

        d1 = d1.to(device)
        d2 = d2.to(device)
        d1_edge = d1_edge.to(device)
        d2_edge = d2_edge.to(device)
        weight = weight.to(device)

        output = model(d1, d2, d1_edge, d2_edge)
        
        y = d1.y.reshape(-1, output.shape[1]).to(device).float()

        loss = criterion(output, y)
        # loss = binary_cross_entropy(output, y, weight.unsqueeze(-1), True)
        
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
        
    return (sum(total_loss) / len(total_loss))

def evaluate_regression(model, data, criterion, device):
    model.eval()
    total_loss = []
    y_predict = []
    y_pred = []
    y_test = []
    with torch.no_grad():
        for d1, d2, d1_edge, d2_edge, _, _ in data:

            d1 = d1.to(device)
            d2 = d2.to(device)
            d1_edge = d1_edge.to(device)
            d2_edge = d2_edge.to(device)

            output = model(d1, d2, d1_edge, d2_edge)
                        
            y = d1.y.reshape(-1, output.shape[1]).to(device).float()            
            total_loss.append(criterion(output,  y).item())
            output = output.cpu().detach().numpy()
            
            y_pred.append(output)
            y_test.append(y.cpu().numpy())

    y_pred = np.array(y_pred).flatten()
    y_test = np.array(y_test).flatten()
    print(y_pred)
    print(y_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    print(f'MAE:{mae}')
    print(f'MSE:{mse}')
    print(f'R2:{r2}')
    
    model.train()
    return sum(total_loss) / len(total_loss), mae, mse, r2, y_test, y_pred


def do_compute_metrics(probas_pred, target):

    p, r, t = precision_recall_curve(target, probas_pred)
    auc_prc = auc(r, p)

    all_F_measure = np.zeros(len(p))
    for k in range(0, len(p)):
        if (p[k] + p[k]) > 0:
            all_F_measure[k] = 2 * p[k] * r[k] / (p[k] + r[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = p[max_index]

    fpr, tpr, auc_thresholds = roc_curve(target, probas_pred)
    auc_score = auc(fpr, tpr)

    pred = (probas_pred >= 0.5).astype(int)

    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    acc = accuracy_score(target, pred)
    auc_roc = roc_auc_score(target, probas_pred)
    f1 = f1_score(target, pred)

    print("new auc_score:", auc_score)
    print('new accuracy:', acc)
    print("new precision:", precision)
    print("new recall:", recall)
    print("new f1:", f1)
    print("new auprc_score:", auc_prc)
    print("===================")
    return acc, auc_roc, auc_prc, f1, precision, recall

def evaluate_test_scros(model, data, criterion, device):
    model.eval()
    total_loss = []
    y_predict = []
    y_test = []
    
    with torch.no_grad():
        feature, A, y = data
        if len(y.shape) == 3:
            y = y.squeeze(1)
        feature, A, y = feature.to(device), A.to(device), y.to(device)
        output,_ = model(feature,A)
        total_loss.append((criterion(output,  y)).item())
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        for i in output:
            y_predict.append(i)
        for j in y:
            y_test.append(j)
    y_test = pd.DataFrame(y_test)
    y_predict = pd.DataFrame(y_predict)

    if y_test.shape[1]==1:
        fpr, tpr, threshold = roc_curve(y_test, y_predict)
        AUC = auc(fpr, tpr)
        output_tran = []
        for x in y_predict[0]:
            if x > 0.5:
                output_tran.append(1)
            else:
                output_tran.append(0)
        precision = precision_score(y_test, output_tran)
        recall = recall_score(y_test, output_tran)
    else:
        AUC_all = []
        precision_all = []
        recall_all = []
        for i in range(y_test.shape[1]):
            if max(y_test[i])==0 or max(y_predict[i])==0:
                continue
            fpr, tpr, threshold = roc_curve(y_test[i], y_predict[i])
            AUC = auc(fpr, tpr)
            output_tran = []
            for x in y_predict[i]:
                if x > 0.5:
                    output_tran.append(1)
                else:
                    output_tran.append(0)
            precision = precision_score(y_test[i], output_tran)
            recall = recall_score(y_test[i], output_tran)
            AUC_all.append(AUC)
            precision_all.append(precision)
            recall_all.append(recall)
        AUC = np.mean(AUC_all)
        precision = np.mean(precision_all)
        recall = np.mean(recall_all)

    return (sum(total_loss) / len(total_loss)),AUC,precision,recall