import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, text_ids):

        batch_size = embeddings.size(0)

        # similarity matrix of the pressed socio-demo embeddings (for later usage as a weight)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # mask matrix: places with 1 for same comment_ids, 0 for different comment_ids
        text_same_mask = (text_ids.unsqueeze(1) == text_ids.unsqueeze(0)).float()

        # postive case: places with 1 for same targets, 0 for different targets
        pos_mask = text_same_mask * (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # remove diagnal values
        pos_mask = pos_mask * (1 - torch.eye(batch_size, device=embeddings.device))
        
        # loss for postive examples (same comment, same target): penalize if socio-embedding not similarity enough  
        pos_loss = -torch.sum(F.log_softmax(similarity_matrix, dim=1) * pos_mask) / max(torch.sum(pos_mask), 1)

        # negtive case: places with 1 for different targets, 0 for similar targets
        neg_mask = text_same_mask * (1 - (labels.unsqueeze(1) == labels.unsqueeze(0)).float())

        # loss for negtive examples(same comment, but different target) penalize if socio-embeddings are similar
        penalty_loss = torch.sum(F.softmax(similarity_matrix, dim=1) * neg_mask) / max(torch.sum(neg_mask), 1)

        return pos_loss + penalty_loss


class Contrastive_Combined_Loss(nn.Module):
    def __init__(self, bce_weight=1.0, set_contrastive_weight=1.0, temperature=0.07):
        super().__init__()
        self.bce_weight = bce_weight
        self.set_contrastive_weight = set_contrastive_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = Contrastive_Loss(temperature=temperature)

    def forward(self, classification_logits, contrastive_emb, targets, labels, text_ids ):

        bce_loss = self.bce_loss(classification_logits.squeeze(), targets.float())
        
        contrastive_loss = self.contrastive_loss(contrastive_emb, labels, text_ids)

        total_loss = (self.bce_weight * bce_loss +
                     self.set_contrastive_weight * contrastive_loss)

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

        return masked_loss.sum() / (mask.sum() + 1e-8)
    

