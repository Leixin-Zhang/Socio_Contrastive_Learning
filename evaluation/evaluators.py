import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def simple_evaluator(model, dataloader, threshold=0.5, get_preds=False):
    model.eval()

    all_preds = []
    all_targets = []
    all_prob_preds = []

    with torch.no_grad():
        for batch in dataloader:

            inputs, batch_targets = batch[:-1], batch[-1]
            outputs = model(*inputs)

            prob_preds = torch.sigmoid(outputs).squeeze()
            preds = (prob_preds > threshold).int()
            
            all_prob_preds.extend(prob_preds.numpy())
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_targets.numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    
    auc = roc_auc_score(all_targets, all_prob_preds)


    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    if get_preds == True:
        return metrics, all_prob_preds
    
    return metrics



def multi_task_evaluator(model, dataloader, threshold=0.5, get_preds = False):


    """
    evaluator：evaluate labels from specific annotators (with annotator_id) select specific anntoator for each item
    
    Args:
        model: multi-task model to be evaluated
        test_dataloader: data loader for test data, including / Annotator Index in Multi-task Head/ text tensors/ & /test labels/
    """

    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for annotator_id, x, y in dataloader:

            logits = model(x)                 
            probs = torch.sigmoid(logits)
            idx = annotator_id.unsqueeze(1)
            prob = probs.gather(1, idx).squeeze(1)

            pred = (prob > threshold).float()

            all_probs.append(prob)
            all_preds.append(pred)
            all_targets.append(y)

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    auc = roc_auc_score(y_true,y_prob)
    
    metrics =  {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    if get_preds == True:
        return metrics, y_pred
    
    return metrics


