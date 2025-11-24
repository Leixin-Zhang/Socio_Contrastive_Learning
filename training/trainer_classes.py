import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model: nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    criterion: Callable  
    optimizer: torch.optim.Optimizer
    eval_func: Callable  
    num_epochs: int = 10
    model_type: str = 'simple'  
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Simple_Model_Trainer:

    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        config.model.to(config.device)
        history = {
            'train_loss': [],
            'val_metrics': []
        }
        for epoch in range(config.num_epochs):

            config.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for text,label in config.train_loader:
                config.optimizer.zero_grad()
                predictions = config.model(text)
                loss = config.criterion(predictions, label)
                loss.backward()
                config.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            train_epoch_loss = epoch_loss / num_batches 
      
            history['train_loss'].append(train_epoch_loss)
            if config.val_loader != None:
                history['val_metrics'].append(config.eval_func(config.model,config.val_loader))
            
        return history
 
class Multi_Task_Trainer:
        
    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        config.model.to(config.device)
        history = {
            'train_loss': [],
            'val_metrics': []
        }
   

        for epoch in range(config.num_epochs):

            config.model.train()

            epoch_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_targets, batch_mask in config.train_loader:
                config.optimizer.zero_grad()
                predictions = config.model(batch_features, batch_mask)
                loss = config.criterion(predictions, batch_targets, batch_mask)
                loss.backward()
                config.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            train_epoch_loss = epoch_loss / num_batches 
      
            history['train_loss'].append(train_epoch_loss)
            if config.val_loader != None:
                history['val_metrics'].append(config.eval_func(config.model,config.val_loader))
            
        return history
    
  
class Socio_Feature_Trainer:
          
    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        config.model.to(config.device)
        history = {
            'train_loss': [],
            'val_metrics': []
        }

        for epoch in range(config.num_epochs):

            config.model.train()

            epoch_loss = 0.0
            num_batches = 0

            for (social, text, labels) in config.train_loader:
                config.optimizer.zero_grad()
                predictions = config.model(social, text)
                loss = config.criterion(predictions, labels)
                loss.backward()
                config.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            train_epoch_loss = epoch_loss / num_batches 
      
            history['train_loss'].append(train_epoch_loss)
            if config.val_loader != None:
                history['val_metrics'].append(config.eval_func(config.model,config.val_loader))
        return history
        
    
class Contrastive_Trainer:

    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        config.model.to(config.device)
        history = {
            'train_loss': [],
            'val_metrics': []
        }

        for epoch in range(config.num_epochs):

            config.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_social, batch_text, batch_targets, batch_comment_ids in config.train_loader:
                config.optimizer.zero_grad()

                classification_logits, contrastive_emb = config.model(batch_social,batch_text, return_contrastive=True)

                loss_dict = config.criterion(
                    classification_logits=classification_logits,
                    contrastive_emb=contrastive_emb,
                    targets=batch_targets,
                    labels=batch_targets.long(),
                    text_ids=batch_comment_ids
                )

                loss_dict['total_loss'].backward()
                 # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                config.optimizer.step()

            epoch_loss += loss_dict['total_loss'].item()

            # precision, recall, f1 = config.eval_func(config.model, config.val_loader)
            # print(f'Val - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            history['train_loss'].append( epoch_loss/len(config.train_loader))
        
            if config.val_loader != None:
                    history['val_metrics'].append(config.eval_func(config.model, config.val_loader))

        return history





