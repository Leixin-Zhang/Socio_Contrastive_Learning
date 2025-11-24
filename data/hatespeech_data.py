import datasets
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class HateSpeechDatasetLoader:
    
    def __init__(self):
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")

    def load_and_preprocess_data(self):
        dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
        hate_df = dataset['train'].to_pandas()
        
        # data filter (filter annotator who annotated less than 20, filter comment annotated by less than 2 annotators)
        selected_hate_df = hate_df.groupby('annotator_id').filter(lambda x: len(x) >= 20)
        selected_hate_df = selected_hate_df.groupby('comment_id').filter(lambda x: len(x) >= 2)
        
        # fill na, change value type
        final_hate_df = selected_hate_df.fillna(0)
        bool_col = selected_hate_df.select_dtypes(include=['bool']).columns
        final_hate_df[bool_col] = final_hate_df[bool_col].astype(int)

        # create 0 / 1 as binary labels (non-hate = 0, hate = 1 )
        final_hate_df = final_hate_df.assign(binary_hatespeech=final_hate_df['hatespeech'])
        final_hate_df['binary_hatespeech'] = final_hate_df['binary_hatespeech'].replace(2, 1)

        return final_hate_df
    

    def get_text_tensor(self, df, comment_ids, embeddings, mapping):

        filtered_df = df[df.comment_id.isin(comment_ids)]
        embedding_list = [embeddings[mapping[comment_id]] for comment_id in filtered_df.comment_id.values]
        return torch.tensor(np.array(embedding_list), dtype=torch.float32)


    def get_target_tensor(self, df, comment_ids, target_column='binary_hatespeech'):

        filtered_df = df[df.comment_id.isin(comment_ids)]
        return torch.tensor(filtered_df[target_column].values, dtype=torch.float32)

    def get_unique_comment_ids(self, df):
        unique_df = df.drop_duplicates(subset=['comment_id'])
        return unique_df.comment_id.tolist()
    
    def get_unique_text_embeddings(self, df):
        unique_df = df.drop_duplicates(subset=['comment_id'])
        text_list = unique_df.text.tolist()
        return self.sbert.encode(text_list)

    def comment_mapping_dict(self, df):
        unique_comment_ids = self.get_unique_comment_ids(df)
        return dict(zip(unique_comment_ids, range(len(unique_comment_ids))))

    # data preparation for majority voted labels
    def get_aggregated_data(self, df, comment_ids, text_embeddings):
            
            mapping = self.comment_mapping_dict(df)
            aggregated_df = (df[df.comment_id.isin(comment_ids)]
                            .groupby('comment_id')[['hatespeech']]
                            .agg(lambda x: (x.mean() >= 0.5).astype(int))
                            .reset_index()
                            .rename(columns={'hatespeech': 'aggregated_hatespeech'}))
            
            text_tensor = torch.tensor(np.array([text_embeddings[mapping[i]] for i in aggregated_df.comment_id.values]),dtype=torch.float32)            
            target_tensor = torch.tensor(aggregated_df.aggregated_hatespeech.values, dtype=torch.float32)
            
            return text_tensor, target_tensor
    

    def get_one_hot_encoding(self, selected_hate_df):
        
        start_idx = selected_hate_df.columns.get_loc('annotator_gender_men')
        end_idx = selected_hate_df.columns.get_loc('annotator_sexuality_other')
        one_hot_df = selected_hate_df.iloc[:, start_idx: end_idx+1]
        
        return torch.tensor(one_hot_df.to_numpy(), dtype=torch.float32)
    
    def get_socio_embedding_tensor(self, df):
        annotator_embedding_mapping = self.annotator_embedding_dict(df)
        annotator_embeddings = torch.tensor(np.array([annotator_embedding_mapping[i] for i in df.annotator_id.values]), 
                                           dtype = torch.float32)
        return annotator_embeddings
    
    def annotator_embedding_dict(self, hate_df):
        
        gender_array = hate_df.apply(lambda row: 'gender: ' + row['annotator_gender'] + ', transgender' if row['annotator_transgender'] == True else 'gender: ' + row['annotator_gender'] , axis=1).values
        age_array = hate_df.apply(lambda row: str(int(row['annotator_age'])) + ' years old' if row['annotator_age']==row['annotator_age'] else 'age: None' , axis = 1).values
        edu_array = hate_df.apply(lambda row: 'education level: ' + row['annotator_educ'].replace('_', ' '), axis = 1).values
        income_array = hate_df.apply(lambda row:  'income: ' + str(row['annotator_income']) if row['annotator_income'] is not None else 'income: NONE', axis = 1).values
        ideology_array = hate_df.apply(lambda row: 'ideology: ' + row['annotator_ideology'], axis = 1).values
        
        religion = []
        religion_start_idx = hate_df.columns.to_list().index('annotator_religion_atheist')
        religion_end_idx = hate_df.columns.to_list().index('annotator_religion_other')
        for row_idx in range(len(hate_df)):
            religion_per_person = 'religion: '
            for col_idx in range(religion_start_idx, religion_end_idx):
                if hate_df.iloc[row_idx,col_idx] != 0:
                    religion_per_person += hate_df.columns[col_idx][20:] + ' '
                religion.append(religion_per_person.strip())

        race = []
        race_start_idx = hate_df.columns.to_list().index('annotator_race_asian')
        race_end_idx = hate_df.columns.to_list().index('annotator_race_other')
        for row_idx in range(len(hate_df)):
            race_per_person = 'race: '
            for col_idx in range(race_start_idx, race_end_idx):
                if hate_df.iloc[row_idx,col_idx] != 0:
                    race_per_person += hate_df.columns[col_idx][15:] + ' '
                race.append(race_per_person.strip())

        sexuality = []
        sexuality_start_idx = hate_df.columns.to_list().index('annotator_sexuality_bisexual')
        sexuality_end_idx = hate_df.columns.to_list().index('annotator_sexuality_other')
        for row_idx in range(len(hate_df)):
            sexuality_per_person = 'sexuality: '
            for col_idx in range(sexuality_start_idx, sexuality_end_idx):
                if hate_df.iloc[row_idx,col_idx] != 0:
                    sexuality_per_person += hate_df.columns[col_idx][20:] + ' '
                sexuality.append(sexuality_per_person.strip())

        ## 8 factors: gender, age, educ, income, ideology, religion, race, sexuality
        ## annotator_id: annotator_embedding

        annotator_embedding_dic = {}
        for df_id, annotator_id in enumerate(hate_df.annotator_id.values):
            if annotator_id not in annotator_embedding_dic.keys():
                annotator_embedding_dic[annotator_id] = self.sbert.encode([gender_array[df_id],age_array[df_id],edu_array[df_id],income_array[df_id],
                                    ideology_array[df_id],religion[df_id],race[df_id],sexuality[df_id]])
        
        return annotator_embedding_dic


 
