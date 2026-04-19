
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from data_processing.toxicity_data_processing import ToxicDatasetLoader,ToxicFeatureBuilder
from data_processing.hatespeech_data_processing import HateSpeechDatasetLoader,HateSpeechFeatureBuilder
from data_processing.dataset_loader import simple_dataloader, multi_task_dataloader, annotator_feature_dataloader, contrastive_dataloader
from models.baseline_models import Multi_Task_Model,Simple_Model
from models.socio_feature_model import Socio_Feature_Model
from models.contrastive_model import Contrastive_Model
from training.self_defined_loss import MaskedBCELoss, Contrastive_Combined_Loss
from training.trainer_classes import TrainingConfig,GenericTrainer,Multi_Task_Trainer,Contrastive_Trainer
from evaluation.evaluators import  multi_task_evaluator, simple_evaluator




class TrainModels():

    def __init__(self, data = 'toxic', test_proportion = 0.4, encoder = 'SBERT'):
        
        self.data = data

        if data == 'toxic':
            dataset_loader = ToxicDatasetLoader()
            self.feature_builder = ToxicFeatureBuilder(encoder=encoder)

        elif data == 'hatespeech':
            dataset_loader = HateSpeechDatasetLoader()
            self.feature_builder = HateSpeechFeatureBuilder(encoder=encoder)


        df_all = dataset_loader.load_and_preprocess_data().reset_index(drop=True)
        self.pivot_df = dataset_loader.get_pivot_df(df_all)

        # ------------- ------------------------
        # Data Split 
        # -------------------------------------
        unique_comment_ids =df_all.drop_duplicates(subset=['comment_id']).comment_id.tolist()
        print('unique comments: ', len(unique_comment_ids))

        self.train_comment_id, self.test_comment_id =  train_test_split(unique_comment_ids, test_size=test_proportion, random_state=36)
        self.train_dataframe = df_all[df_all.comment_id.isin(self.train_comment_id)]
        self.test_dataframe = df_all[df_all.comment_id.isin(self.test_comment_id)]
        print('train_df size: ' , len(self.train_dataframe),'test_df size: ', len(self.test_dataframe))

        # -------------------------------------
        # Text Embedding
        # -------------------------------------
        self.text_embedding_mapping = self.feature_builder.build_text_embedding_dict(df_all)

        self.train_text, self.train_target = self.feature_builder.build_tensors(df_all, self.train_comment_id, self.text_embedding_mapping)
        self.test_text, self.test_target = self.feature_builder.build_tensors(df_all, self.test_comment_id, self.text_embedding_mapping)
        
        print('train unique: ',len(self.train_comment_id),'test unique: ',len(self.test_comment_id))
        
        # -------------------------------------
        # SOCIO-DEMOGRAPHIC Encoding
        # -------------------------------------
        self.train_socio_one_hot = self.feature_builder.get_one_hot_tensor(df_all,self.train_comment_id)
        self.test_socio_one_hot = self.feature_builder.get_one_hot_tensor(df_all,self.test_comment_id)
        
     

    def train_simple_model(self, epoch=7):
        
        """
        Training with aggregated labels (test with perspective labels)
        """

        # aggregated labels for training dataloader



        train_text_aggregation, train_target_aggregation = self.feature_builder.build_tensor_aggregated_labels(
                                                            self.train_dataframe,self.train_comment_id,self.text_embedding_mapping)
        
        # Dataloader for aggregated label training 
        
        
        train_single_dataloader = simple_dataloader(train_text_aggregation,train_target_aggregation,shuffle=True) 
        test_dataloader = simple_dataloader(self.test_text,self.test_target,shuffle=False)

        single_model = Simple_Model(input_dim = train_text_aggregation.shape[1])

        trainer = GenericTrainer()
        single_model_config = TrainingConfig(
            model=single_model,
            train_loader=train_single_dataloader,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(single_model.parameters(),lr = 0.001),
            eval_func=simple_evaluator,
            num_epochs=epoch,
            model_type='aggregated_model',
            device= 'cpu'
        )

        single_model_history = trainer.train(single_model_config)

        evaluation_result,preds = simple_evaluator(single_model,test_dataloader,get_preds=True)
        print(f'aggregated_model_result: {evaluation_result}')

        detailed_result_df =  pd.DataFrame({'single_model_predictions': preds})

        return evaluation_result,single_model_history, detailed_result_df
    


    def train_multi_task_model(self, epoch=7):

        """
        Davini's Multitask Learning: Predict individual labels with separate head 
                                     for each annotator.
        """

        train_multi_task_dataloader = multi_task_dataloader(self.pivot_df, self.text_embedding_mapping, self.train_comment_id)



        # get index of a specific annotator on multi-task predications, evaluate individually
        annotator_pivot_map = dict(zip(self.pivot_df.columns.values, range(len(self.pivot_df.columns))))
        test_annotator_idx = torch.tensor(np.array([annotator_pivot_map[i] for i in self.test_dataframe.annotator_id.values]),dtype=torch.long)
        test_multi_task_dataloader = annotator_feature_dataloader(test_annotator_idx,self.test_text,self.test_target, shuffle=False)

        multi_task_model = Multi_Task_Model(input_dim = self.train_text.shape[1], num_annotators=len(self.pivot_df.columns))

        multi_task_config = TrainingConfig(
            model=multi_task_model,
            train_loader=train_multi_task_dataloader,
            criterion=MaskedBCELoss(),
            optimizer=torch.optim.Adam(multi_task_model.parameters(), lr=0.001),
            eval_func=multi_task_evaluator,
            num_epochs=epoch,
            model_type='multi_task_model',
        )
        trainer = Multi_Task_Trainer()
        multi_task_history = trainer.train(multi_task_config)
        evaluation_result,preds = multi_task_evaluator(multi_task_model,test_multi_task_dataloader,get_preds=True)
        detailed_result_df =  pd.DataFrame({'multi_task_predictions': preds})

        print(f'multi_task_result: {evaluation_result}')


        return evaluation_result,multi_task_history,detailed_result_df

    def train_one_hot_model(self,epoch=7):

        train_one_hot_dataloader = annotator_feature_dataloader(self.train_socio_one_hot,self.train_text,self.train_target,shuffle=True)
        test_one_hot_dataloader = annotator_feature_dataloader(self.test_socio_one_hot,self.test_text,self.test_target,shuffle=False)

        trainer = GenericTrainer()
        dim = self.train_socio_one_hot.shape[1] + self.train_text.shape[1]
        one_hot_model = Socio_Feature_Model(input_dim=dim)

        one_hot_model_config  =  TrainingConfig(
            model=one_hot_model,
            train_loader=train_one_hot_dataloader,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(one_hot_model.parameters(),lr = 0.001),
            eval_func=simple_evaluator,
            num_epochs=epoch,
            model_type='one_hot_model'
        )
        one_hot_model_history = trainer.train(one_hot_model_config)
        one_hot_model_result,preds = simple_evaluator(one_hot_model,test_one_hot_dataloader,get_preds=True)
        
        print(f'one_hot_model: {one_hot_model_result}')
        

        detailed_result_df =  pd.DataFrame({'one_hot_model_predictions': preds})

        return one_hot_model_result,one_hot_model_history,detailed_result_df



    def train_social_embedding_model(self,epoch=7):
        
        train_social_embedding_dataloader = annotator_feature_dataloader(
            self.feature_builder.build_socio_embedding_tensor(self.train_dataframe),
            self.train_text,
            self.train_target, shuffle=True
        )

        test_social_embedding_dataloader = annotator_feature_dataloader(
            self.feature_builder.build_socio_embedding_tensor(self.test_dataframe),
            self.test_text,
            self.test_target, shuffle=False
        )

        trainer = GenericTrainer()
        dim = 2 * self.train_text.shape[1]

        socio_embedding_model = Socio_Feature_Model(input_dim=dim)
        socio_embedding_config = TrainingConfig(
            model = socio_embedding_model,
            train_loader=  train_social_embedding_dataloader,
            criterion = nn.BCEWithLogitsLoss(),
            optimizer = torch.optim.Adam(socio_embedding_model.parameters(), lr=0.001),
            eval_func = simple_evaluator,
            num_epochs=epoch,
            model_type = 'social_embedding',
        )

        social_embedding_history = trainer.train(socio_embedding_config)
        result,preds = simple_evaluator(socio_embedding_model,test_social_embedding_dataloader,get_preds=True)
        print(f'social_embedding_model: {result}')

        detailed_result_df =  pd.DataFrame({'socio_embedding_predictions': preds})
        return result,social_embedding_history,detailed_result_df

    def train_contrastive_model(self, epoch=7,contrastive_loss_w = 1):
        
        """
        Contrastive model with 1. Contrastive Loss (for pressed one-hot-encoding projection) 
                            2. BCE lOSS (for final label).
        """
        
        train_contrastive_loader = contrastive_dataloader(
            self.train_socio_one_hot,
            self.train_text,
            self.train_target, 
            torch.tensor(self.train_dataframe.comment_id.values,dtype=torch.float32))

        test_dataloader = annotator_feature_dataloader(self.test_socio_one_hot,self.test_text,self.test_target,shuffle=False)



        trainer = Contrastive_Trainer()
        contrastive_model = Contrastive_Model(socio_dim=self.train_socio_one_hot.shape[1],other_dim=self.train_text.shape[1])

        contrastive_model_config = TrainingConfig(
            model=contrastive_model,
            train_loader=train_contrastive_loader,
            criterion=Contrastive_Combined_Loss(set_contrastive_weight=contrastive_loss_w),
            optimizer=torch.optim.Adam(contrastive_model.parameters(),lr = 0.001),
            eval_func=simple_evaluator,
            num_epochs=epoch,
            model_type='contrastive_model',
        )
        contrastive_model_history = trainer.train(contrastive_model_config)



        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir = f'./experiment_results/saved_{self.data}_models'

        os.makedirs(dir, exist_ok=True)
        filename = os.path.join(dir, f'{timestamp}_contrastive_model_ablation.pth')
        torch.save(contrastive_model.state_dict(), filename)

        result,preds = simple_evaluator(contrastive_model,test_dataloader,get_preds=True)
        print(f'contrastive_result: {result}')
        detailed_result_df =  pd.DataFrame({'contrastive_predictions': preds})

        return result,contrastive_model_history,detailed_result_df
    
