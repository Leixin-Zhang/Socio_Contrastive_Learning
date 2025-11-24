import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_Loss(nn.Module):
    def __init__(self, temperature=0.1, penalty_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.penalty_weight = penalty_weight



    def forward(self, embeddings, labels, text_ids):
        batch_size = embeddings.size(0)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 创建text_id相同的mask
        text_same_mask = (text_ids.unsqueeze(1) == text_ids.unsqueeze(0)).float()

        # 正样本mask：相同text_id且相同标签（排除自身）
        pos_mask = text_same_mask * (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask = pos_mask * (1 - torch.eye(batch_size, device=embeddings.device))

        # 正样本损失
        pos_loss = -torch.sum(F.log_softmax(similarity_matrix, dim=1) * pos_mask) / max(torch.sum(pos_mask), 1)

        # 惩罚项：相同text_id但不同标签
        penalty_mask = text_same_mask * (1 - (labels.unsqueeze(1) == labels.unsqueeze(0)).float())
        penalty_mask = penalty_mask * (1 - torch.eye(batch_size, device=embeddings.device))
        penalty_loss = torch.sum(F.softmax(similarity_matrix, dim=1) * penalty_mask) / max(torch.sum(penalty_mask), 1)

        # return pos_loss + self.penalty_weight * penalty_loss

        return self.penalty_weight * penalty_loss

class Contrasive_Combined_Loss(nn.Module):
    def __init__(self, bce_weight=1.0, contrastive_weight=1, temperature=0.07, penalty_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.contrastive_weight = contrastive_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = Contrastive_Loss(temperature=temperature, penalty_weight=penalty_weight)

    def forward(self, classification_logits, contrastive_emb, targets, labels, text_ids=None):
        bce_loss = self.bce_loss(classification_logits.squeeze(), targets.float())
        contrastive_loss = self.contrastive_loss(contrastive_emb, labels, text_ids)

        total_loss = (self.bce_weight * bce_loss +
                     self.contrastive_weight * contrastive_loss)

        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'contrastive_loss': contrastive_loss
        }

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets, mask):

        loss_per_element = self.bce(predictions, targets)

        masked_loss = loss_per_element * mask
        valid_losses = mask.sum()

        if valid_losses > 0:
            return masked_loss.sum() / valid_losses
        else:
            return masked_loss.sum() * 0
