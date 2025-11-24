
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split

from data.hatespeech_data import HateSpeechDatasetLoader
from data.dataset_loader import simple_dataloader, multi_task_dataloader, socio_feature_dataloader,contrastive_dataloader

from models.baseline_models import MultiTaskModel,Simple_Model
from models.one_hot_model import OneHot_SocialFeature_Model
from models.social_embedding_models import SimpleFusionModel
from models.contrastive_model import Contrasive_Model

from training.self_defined_loss import MaskedBCELoss, Contrasive_Combined_Loss
from training.trainer_classes import TrainingConfig,Socio_Feature_Trainer,Multi_Task_Trainer, Simple_Model_Trainer, Contrastive_Trainer
from evaluation.evaluators import multi_label_evaluator, single_label_evaluator, socio_feature_evaluator


# from utils.config import TRAINING_CONFIG

hate_speech = HateSpeechDatasetLoader()
hate_df = hate_speech.load_and_preprocess_data()


print(len(hate_df))

unique_comment_ids = hate_speech.get_unique_comment_ids(hate_df)
train_comment_id, test_comment_id =  train_test_split(unique_comment_ids, test_size=0.3, random_state=36)
text_embedding_list = hate_speech.get_unique_text_embeddings(hate_df)
comment_mapping_dict = hate_speech.comment_mapping_dict(hate_df)

train_hate_df = hate_df[hate_df.comment_id.isin(train_comment_id)]
test_hate_df = hate_df[hate_df.comment_id.isin(test_comment_id)]

train_text_tensor = hate_speech.get_text_tensor(hate_df,train_comment_id,text_embedding_list,comment_mapping_dict)
test_text_tensor = hate_speech.get_text_tensor(hate_df,test_comment_id,text_embedding_list,comment_mapping_dict)

train_target = hate_speech.get_target_tensor(hate_df,train_comment_id)
test_target = hate_speech.get_target_tensor(hate_df,test_comment_id)

train_socio_one_hot = hate_speech.get_one_hot_encoding(train_hate_df)
test_socio_one_hot = hate_speech.get_one_hot_encoding(test_hate_df)

def train_single_model():
    
    """
    Training with aggregated labels (test with perspective labels)
    """

    # aggregated labels for training dataloader

    train_unique_text_tensor, train_aggregated_target = hate_speech.get_aggregated_data(
                                                        train_hate_df,train_comment_id,text_embedding_list)
    
    # Dataloader for aggregated label training 
    
    train_single_dataloader = simple_dataloader(train_unique_text_tensor,train_aggregated_target) 
    test_single_dataloader = simple_dataloader(test_text_tensor,test_target)

    single_model = Simple_Model()

    trainer = Simple_Model_Trainer()
    single_model_config = TrainingConfig(
        model=single_model,
        train_loader=train_single_dataloader,
        val_loader=test_single_dataloader,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(single_model.parameters(),lr = 0.001),
        eval_func=single_label_evaluator,
        num_epochs=8,
        model_type='test_single_dataloader',
        device= 'cpu'
    )

    single_model_history = trainer.train(single_model_config)
    evaluation_result = single_label_evaluator(single_model,test_single_dataloader)
    print(f'single_model: {evaluation_result}')
    return single_model_history

def train_multi_task_model():

    """
    Davini's Multitask Learning: Predict individual labels altogether 
                                 with each annotator have a separate head
    """

    train_multi_task_dataloader,test_multi_task_dataloader = multi_task_dataloader(
        hate_df, text_embedding_list, train_comment_id, test_comment_id,comment_mapping_dict
    )

    multi_task_model = MultiTaskModel()

    multi_task_config = TrainingConfig(
        model=multi_task_model,
        train_loader=train_multi_task_dataloader,
        val_loader=test_multi_task_dataloader,
        criterion=MaskedBCELoss(),
        optimizer=torch.optim.Adam(multi_task_model.parameters(), lr=0.001),
        eval_func=multi_label_evaluator,
        num_epochs=5,
        model_type='multi_task',
        device= 'cpu'
    )
    trainer = Multi_Task_Trainer()
    multi_task_history = trainer.train(multi_task_config)
    result = multi_label_evaluator(multi_task_model,test_multi_task_dataloader)
    print(f'multi_task_result: {result}')
    return multi_task_history

def train_one_hot_model():

    train_one_hot_dataloader = socio_feature_dataloader(train_socio_one_hot,train_text_tensor,train_target)
    test_one_hot_dataloader = socio_feature_dataloader(test_socio_one_hot,test_text_tensor,test_target)

    trainer = Socio_Feature_Trainer()
    one_hot_model = OneHot_SocialFeature_Model()
    one_hot_model_config  =  TrainingConfig(
        model=one_hot_model,
        train_loader=train_one_hot_dataloader,
        val_loader=test_one_hot_dataloader,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(one_hot_model.parameters(),lr = 0.001),
        eval_func=socio_feature_evaluator,
        num_epochs=3,
        model_type='one_hot_model',
        device=None
    )
    one_hot_model_history = trainer.train(one_hot_model_config)
    one_hot_model_result = socio_feature_evaluator(one_hot_model,test_one_hot_dataloader)
    print(f'one_hot_model: {one_hot_model_result}')
    return one_hot_model_history

def train_social_embedding_model():

    train_social_embedding_dataloader = socio_feature_dataloader(
        hate_speech.get_socio_embedding_tensor(train_hate_df),
        train_text_tensor,
        train_target
    )

    test_social_embedding_dataloader = socio_feature_dataloader(
        hate_speech.get_socio_embedding_tensor(test_hate_df),
        test_text_tensor,
        test_target
    )
    trainer = Socio_Feature_Trainer()

    socio_embedding_model = SimpleFusionModel()
    socio_embedding_config = TrainingConfig(
        model = socio_embedding_model,
        train_loader=  train_social_embedding_dataloader,
        val_loader= test_social_embedding_dataloader, 
        criterion = nn.BCEWithLogitsLoss(),
        optimizer = torch.optim.Adam(socio_embedding_model.parameters(), lr=0.001),
        eval_func = socio_feature_evaluator,
        num_epochs=5,
        model_type = 'social_embedding',
        device = None  
    )

    social_embedding_history = trainer.train(socio_embedding_config)
    result = socio_feature_evaluator(socio_embedding_model,test_social_embedding_dataloader)
    print(f'multi_task_result: {result}')
    
    return social_embedding_history

def train_contrastive_model():
    
    """
    Contrastive model with 1. Contrastive Loss (for pressed one-hot-encoding projection) 
                           2. BCE lOSS (for final label).
    """
    
    train_contrastive_loader, test_contrastive_loader = contrastive_dataloader(
        train_socio_one_hot,train_text_tensor,train_target, 
        torch.tensor(train_hate_df.comment_id.values,dtype=torch.float32),
        test_socio_one_hot,test_text_tensor,test_target, 
        torch.tensor(test_hate_df.comment_id.values,dtype=torch.float32))

    trainer = Contrastive_Trainer()
    contrastive_model = Contrasive_Model()

    contrastive_model_config = TrainingConfig(
        model=contrastive_model,
        train_loader=train_contrastive_loader,
        val_loader=test_contrastive_loader,
        criterion=Contrasive_Combined_Loss(),
        optimizer=torch.optim.AdamW(contrastive_model.parameters(),lr = 0.001),
        eval_func=socio_feature_evaluator,
        num_epochs=5,
        model_type='contrastive_model',
        device=None
    )
    contrastive_model_history = trainer.train(contrastive_model_config)
    result = socio_feature_evaluator(contrastive_model,test_contrastive_loader)
    print(f'contrastive_result: {result}')
    return contrastive_model_history

def run_all_models():

    """
    Call each model's training function and collect the history and evaluation results.
    """
    results = {}
    # Model 1: Simple Model

    single_model_history = train_single_model()
    results['single_model'] = single_model_history

    # Model 2: Multi-Task Model
    multi_task_history = train_multi_task_model()
    results['multi_task_model'] = multi_task_history

    # Model 3: One-Hot Model
    one_hot_history = train_one_hot_model()
    results['one_hot_model'] = one_hot_history

    # Model 4: Social Embedding Model
    social_embedding_history = train_social_embedding_model()
    results['social_embedding_model'] = social_embedding_history

    # Model 5: Contrastive Model
    contrastive_model_history = train_contrastive_model()
    results['contrastive_model'] = contrastive_model_history

    return results

def save_result_to_txt(filename: str = None):
    print('Running starts')
    result = run_all_models()

    """Save training history to a text file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_history_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("TRAINING HISTORY REPORT 0.3-split random_state=0.36\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, result in result.items():
        # Model information
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model Type: {model_name}\n")
            
            # Detailed epoch results
            f.write("DETAILED EPOCH RESULTS:\n")
            f.write("-" * 30 + "\n")
            # for loss, metrics in result.keys():

            for i in range(len(result['train_loss'])):
                f.write(f'epoch {i+1}' "-" + "\n")
                f.write(f'loss {result['train_loss'][i]} \n')
                f.write(f'metrics {result['val_metrics'][i]} \n')
            # train_losses = history.get('train_loss', [])
            # val_metrics = history.get('val_metrics', [])
            

if __name__ == "__main__":
    # Running all models and collecting the results
    save_result_to_txt()


    # Print the results
    # for model_name, result in all_model_results.items():
    #     print(f"Results for {model_name}:")
    #     print(f"History: {result}")
    #     print("----------------------------")


