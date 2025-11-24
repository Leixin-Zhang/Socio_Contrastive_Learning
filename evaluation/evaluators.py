import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support

import torch

# def socio_feature_evaluator(model, test_loader, device='cpu'):

#     model.eval()
#     model.to(device)

#     all_predictions = []
#     all_labels = []

#     with torch.no_grad():
#         for social, text, labels in test_loader:

#             social = social.to(device)
#             text = text.to(device)
#             labels = labels.to(device)

#             logits, _ = model(social, text)

#             probabilities = torch.sigmoid(logits)
#             predictions = (probabilities > 0.5).float()

#             all_predictions.extend(predictions.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     all_predictions = np.array(all_predictions)
#     all_labels = np.array(all_labels)

#     precision = precision_score(all_labels, all_predictions, zero_division=0)
#     recall = recall_score(all_labels, all_predictions, zero_division=0)
#     f1 = f1_score(all_labels, all_predictions, zero_division=0)

#     metrics = {
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#     }
#     return metrics


def multi_label_evaluator(model, test_dataloader, threshold=0.5):
    
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch_features, batch_targets, batch_mask in test_dataloader:
            batch_features = batch_features
            batch_mask = batch_mask

            predictions = model(batch_features, batch_mask)  
            probabilities = torch.sigmoid(predictions)
            binary_preds = (probabilities > threshold).float() 

            all_preds.append(binary_preds)
            all_targets.append(batch_targets)
            all_masks.append(batch_mask)

    
    binary_preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    binary_preds_flat = binary_preds.flatten()
    targets_flat = targets.flatten()
    mask_flat = masks.flatten()

    valid_mask = mask_flat != 0
    valid_predictions = binary_preds_flat[valid_mask]
    valid_targets = targets_flat[valid_mask]
    precision, recall, fscore, _ = precision_recall_fscore_support(
        valid_targets.numpy(), valid_predictions.numpy(),average='binary'  
    )
 
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': fscore,
    }
    return metrics


def single_label_evaluator(model, test_loader, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device).float()
            outputs = model(data)
            probabilities = torch.sigmoid(outputs) if outputs.dim() > 0 else torch.sigmoid(outputs.unsqueeze(0))

            # 获取预测结果（阈值0.5）
            predictions = (probabilities > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)

    # 计算各种指标
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    return metrics


def socio_feature_evaluator(model, dataloader, threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_social, batch_features, batch_targets in dataloader:
            outputs = model(batch_social,batch_features)
            preds = torch.sigmoid(outputs).squeeze()
            preds = (preds > threshold).int()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    return metrics