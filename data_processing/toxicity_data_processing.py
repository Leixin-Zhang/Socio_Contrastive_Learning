import pandas as pd
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from data_processing.text_encoder import BERT_CLS, BERT_MeanPooling, RoBERTa_MeanPooling


class ToxicDatasetLoader:

    def load_and_preprocess_data(self):
        json_path = Path(__file__).parent / "toxicity_dataset.json"
        df = pd.read_json(json_path, lines=True)

        df = (
            df.explode("ratings")
              .reset_index(drop=True)
              .drop(columns=["source"])
              .join(pd.json_normalize(df.explode("ratings")["ratings"]))
        )

        # remap ids (vectorized)
        df["comment_id"] = df["comment"].map(
            {c: i for i, c in enumerate(df["comment"].unique())}
        )
        df["annotator_id"] = df["worker_id"].map(
            {w: i for i, w in enumerate(df["worker_id"].unique())}
        )

        df = df.drop_duplicates(["annotator_id", "comment_id"])

        # binary label (vectorized)
        df["binary_toxic_score"] = (df["toxic_score"] != 0).astype(int)

        # filtering
        df = df.groupby("annotator_id").filter(lambda x: len(x) > 20)
        df = df.groupby("comment_id").filter(lambda x: len(x) > 4)

        return df

    def get_pivot_df(self, result_df):
        return result_df.pivot(
            index="comment_id",
            columns="annotator_id",
            values="binary_toxic_score"
        )
    


class ToxicFeatureBuilder:
    
    """
    Class for:
    - Text embeddings
    - Socio-demographic embeddings
    - Tensor Construction
    """
    
    def __init__(self, encoder="SBERT"):

        if encoder == "SBERT":
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        elif encoder == "BERT_CLS":
            self.encoder = BERT_CLS()
        elif encoder == "BERT_MeanPooling":
            self.encoder = BERT_MeanPooling()
        elif encoder == 'RoBERTa_MeanPooling':
            self.encoder = RoBERTa_MeanPooling()

        else:
            raise ValueError(f"Unknown encoder: {encoder}")


    # ------------------------------------------------------
    # TEXT EMBEDDING MAPPING (comment-id to text embeddings)
    # ------------------------------------------------------
    def build_text_embedding_dict(self, df):

        unique_df = df.drop_duplicates("comment_id")
        texts = unique_df.comment.tolist()

        embeddings = self.encoder.encode(texts, batch_size=32)

        return dict(zip(unique_df.comment_id, embeddings))

    # ---------------------------
    # INPUT-OUTPUT ALIGNMENT
    # ---------------------------
    
    def build_tensors(self, df, comment_ids, embedding_dict):

        df_sub = df[df.comment_id.isin(comment_ids)]

        text_tensor = torch.stack([
            torch.as_tensor(embedding_dict[cid], dtype=torch.float32) 
            for cid in df_sub.comment_id.values])

        target_tensor = torch.tensor(
            df_sub['binary_toxic_score'].values,
            dtype=torch.float32
        )

        return text_tensor, target_tensor
    

    # ---------------------------------------------------
    # AGGREGATED LABELS (optional baseline: simple model)
    # ---------------------------------------------------

    
    def build_tensor_aggregated_labels(self, df, comment_ids, embedding_dict):

        df_sub = df[df.comment_id.isin(comment_ids)]

        # get majority voted labels
        aggregated = (
            df_sub.groupby("comment_id")["toxic_score"]
            .mean()
            .gt(0.75)
            .astype(int)
            .reset_index(name="aggregated_toxic_score")
        )


        text_tensor = torch.stack([
            torch.as_tensor(embedding_dict[cid], dtype=torch.float32) 
            for cid in aggregated.comment_id.values])



        target_tensor = torch.tensor(
            aggregated["aggregated_toxic_score"].values,
            dtype=torch.float32
        )

        return text_tensor, target_tensor


    # -------------------------------------
    # SOCIO-DEMOGRAPHIC ONE-HOT EMBEDDING
    # -------------------------------------

    def get_one_hot_tensor(self, result_df, selected_comment_id):

        race_dummies = result_df['race'].str.get_dummies(sep=',').add_prefix('race_')

        # socio demo columns
        socio_demo_columns = [
            'gender',
            'identify_as_transgender',
            'education',
            'age_range',
            'lgbtq_status',
            'political_affilation',
            'is_parent',
            'religion_important'
        ]

        # one-hot encoding
        df_encoded = pd.get_dummies(
            result_df,
            columns=socio_demo_columns,
            prefix=socio_demo_columns
        )

        # concat race features
        df_result = pd.concat([df_encoded, race_dummies], axis=1)
        df_result = df_result[df_result.comment_id.isin(selected_comment_id)] 

        start_col = df_result.columns.get_loc('gender_Female')
        one_hot_df = df_result.iloc[:, start_col:].astype(np.float32)

        return torch.tensor(one_hot_df.to_numpy(), dtype=torch.float32)


   

        # # combine
        # df = pd.concat([df, race_dummies], axis=1)

        # # keep only numeric features (clean + avoids fragile column slicing)
        # df = df.select_dtypes(include='number').astype(int)

        # return torch.tensor(df.to_numpy(), dtype=torch.float32)


    # -----------------------------------------
    # SOCIO-DEMOGRAPHIC EMBEDDING FROM ENCODERS
    # -----------------------------------------

    # def build_socio_embedding_tensor(self, df):

    #     # 1 Get annotator embedding dict
    #     annotator_embedding_dict = self.annotator_embedding_dict(df)

    #     # 2️ Align annotator embedding to df order
    #     annotator_embeddings = np.stack([
    #         annotator_embedding_dict[aid] for aid in df.annotator_id.values
    #     ])

    #     # 3 Covert to tensor 
    #     return torch.tensor(annotator_embeddings, dtype=torch.float32)


    def build_socio_embedding_tensor(self, df):

        annotator_embedding_dict = self.annotator_embedding_dict(df)

        annotator_embeddings = torch.stack([
            torch.as_tensor(annotator_embedding_dict[aid], dtype=torch.float32)
            for aid in df.annotator_id.values
        ])
        return annotator_embeddings
    

    def annotator_embedding_dict(self, result_df):

        df = result_df.drop_duplicates("annotator_id")

        def build_text(row):
            gender = (
                f"gender: {row['gender']}"
                + (", transgender" if row["identify_as_transgender"] == "Yes" else "")
            )

            text = ", ".join([
                f"race: {row['race']}",
                gender,
                f"education: {row['education']}",
                f"age: {row['age_range']} years old",
                f"lgbtq: {row['lgbtq_status']}",
                f"political affiliation: {row['political_affilation']}",
                f"parent: {row['is_parent']}",
                f"religion: {row['religion_important']}"
            ])

            return text

        texts = df.apply(build_text, axis=1).tolist()

        embeddings = self.encoder.encode(texts)
        
        return dict(zip(df["annotator_id"], embeddings))



    # def annotator_embedding_dict(self, result_df):

    #     df = result_df.drop_duplicates("annotator_id").copy()

    #     df["race"] = "race: " + df["race"].astype(str)

    #     df["gender"] = np.where(
    #         df["identify_as_transgender"].eq("Yes"),
    #         "gender: " + df["gender"].astype(str) + ", transgender",
    #         "gender: " + df["gender"].astype(str)
    #     )

    #     df["education"] = "education: " + df["education"].astype(str)
    #     df["age"] = "age: " + df["age_range"].astype(str) + " years old"
    #     df["lgbtq"] = df["lgbtq_status"].astype(str)
    #     df["politic"] = "political affilation: " + df["political_affilation"].astype(str)
    #     df["parent"] = "parent: " + df["is_parent"].astype(str)
    #     df["religion"] = "religion: " + df["religion_important"].astype(str)

    #     feature_cols = [
    #         "race", "gender", "education", "age",
    #         "lgbtq", "politic", "parent", "religion"
    #     ]

    #     # flatten all texts 

    #     texts = df[feature_cols].agg(", ".join, axis=1)

    #     embeddings = self.encoder.encode(texts)

    #     return dict(zip(df["annotator_id"], embeddings))






        # df["annotator_text"] = df[feature_cols].agg(", ".join, axis=1)
        # embeddings = self.encoder.encode(texts)

        # annotator_dict = dict(zip(df["annotator_id"], df["annotator_text"]))

        # return annotator_dict
    

    



    # def build_socio_embedding_tensor(self, result_df):
    

    #     # Create a dictionary to cache embeddings and avoid repetitive encoding
    #     embedding_cache = {}

    #     race = result_df.apply(lambda row: 'race: '+ str(row.race), axis=1).values
    #     gender = result_df.apply(lambda row: 'gender:' + str(row.gender) + ', transgender' 
    #                             if row.identify_as_transgender == 'Yes' else 'gender: ' + str(row.gender), axis=1).values
    #     education = result_df.apply(lambda row: 'education:' + str(row.education), axis=1).values
    #     age = result_df.apply(lambda row: 'age:' + str(row.age_range) + ' years old', axis=1).values
    #     lgbtq = result_df.apply(lambda row: str(row.lgbtq_status), axis=1).values 
    #     politic = result_df.apply(lambda row: 'political affilation:' + str(row.political_affilation), axis=1).values
    #     parent = result_df.apply(lambda row: 'parent:' + str(row.is_parent), axis=1).values
    #     religion = result_df.apply(lambda row: 'religion:' + str(row.religion_important), axis=1).values
    #     # Combine all feature lists
    #     feature_lists = [race, gender, education, age, lgbtq, politic, parent, religion]

    #     # Get all unique texts across all features to minimize encoding
    #     all_texts = set()
    #     for feature_list in feature_lists:
    #         all_texts.update(feature_list)


    #     # Pre-compute embeddings for all unique texts
    #     for text in all_texts:
    #         if text not in embedding_cache:
    #             embedding_cache[text] = self.encoder.encode(text, convert_to_tensor=False)

    #     # Create embedded features for each socio-demo feature
    #     embedded_features = []
    #     for i in range(len(race)):
    #         feature_row = []
    #         for feature in feature_lists:
    #             feature_row.append(embedding_cache[feature[i]])
            
            
    #         embedded_features.append(feature_row)

    #     final_embeddings = torch.tensor(np.array(embedded_features),dtype=torch.float32)
        
    #     return final_embeddings

    # # data preparation for majority voted labels
    # def get_aggregated_data(self, df, comment_ids, text_embeddings):
            
    #         mapping = self.comment_mapping_dict(df)
    #         aggregated_df = (df[df.comment_id.isin(comment_ids)]
    #                         .groupby('comment_id')[['toxic_score']]
    #                         .agg(lambda x: (x.mean() >= 0.75).astype(int))
    #                         .reset_index()
    #                         .rename(columns={'toxic_score': 'aggregated_toxic_score'}))
            
    #         text_tensor = torch.tensor(np.array([text_embeddings[mapping[i]] for i in aggregated_df.comment_id.values]),dtype=torch.float32)            
    #         target_tensor = torch.tensor(aggregated_df.aggregated_toxic_score.values, dtype=torch.float32)
            
    #         return text_tensor, target_tensor



    
    # def get_text_tensor(self, df, comment_ids, embeddings, mapping):

    #     filtered_df = df[df.comment_id.isin(comment_ids)]
    #     embedding_list = [embeddings[mapping[comment_id]] for comment_id in filtered_df.comment_id.values]
    #     return torch.tensor(np.array(embedding_list), dtype=torch.float32)


    # def get_target_tensor(self, df, comment_ids, target_column='binary_toxic_score'):

    #     filtered_df = df[df.comment_id.isin(comment_ids)]
    #     return torch.tensor(filtered_df[target_column].values, dtype=torch.float32)

    # def get_unique_comment_ids(self, df):
    #     unique_df = df.drop_duplicates(subset=['comment_id'])
    #     return unique_df.comment_id.tolist()

    # def get_unique_text_embeddings(self, df):
    #     unique_df = df.drop_duplicates(subset=['comment_id'])
    #     text_list = unique_df.comment.tolist()
    #     return self.sbert.encode(text_list)

    # def comment_mapping_dict(self, df): 
    #     unique_comment_ids = self.get_unique_comment_ids(df)
    #     return dict(zip(unique_comment_ids, range(len(unique_comment_ids))))

# class ToxicDatasetLoader:

#     def load_and_preprocess_data(self):
#         current_dir = Path(__file__).parent
#         json_path = current_dir / "toxicity_dataset.json"
#         toxicity_df = pd.read_json(json_path, lines=True)

#         # reorganize pandas df: expand rating columns which contains socio-demo of 5 annotators
#         expanded_df = toxicity_df.explode('ratings').reset_index(drop=True)
#         result_df = pd.concat([expanded_df.drop('ratings', axis=1),
#                             pd.json_normalize(expanded_df['ratings'])],
#                             axis=1).drop('source',axis=1)
        
#         # clean data: re-assign comment_id and annotator_id (# 1 comment_ID corresponds to several comments in original dataset)
#         unique_comment = result_df.comment.unique().tolist()
#         comment_mapping = dict(zip(unique_comment,range(len(unique_comment))))
#         result_df = result_df.drop('comment_id',axis=1)
#         result_df['comment_id'] = result_df.apply(lambda row: comment_mapping[row.comment], axis = 1)
#         worker_id = result_df.worker_id.unique().tolist()
#         annotator_mapping = dict(zip(worker_id,range(len(worker_id))))
#         result_df['annotator_id'] = result_df.apply(lambda row: annotator_mapping[row.worker_id], axis=1)

#         result_df = result_df.drop_duplicates(subset=['annotator_id','comment_id'], keep='first')

#         # create binary_toxic_score column, convert 1-4 to 1 == toxic, 0 == non-toxic
#         result_df['binary_toxic_score'] = result_df.apply(lambda row: 0 if row.toxic_score == 0 else 1, axis=1)
        
#         # filter iems annotated by less than 20 annotators
#         filtered_df = result_df.groupby('annotator_id').filter(lambda x: len(x)>20)
#         filtered_df = filtered_df.groupby('comment_id').filter(lambda x: len(x)>4)
        
#         return filtered_df


    # def get_one_hot_df(self,result_df):

    #     race_dummies = result_df['race'].str.get_dummies(sep=',').add_prefix('race_')
    #     socio_demo_columns = ['gender','identify_as_transgender', 'education', 'age_range', 
    #                         'lgbtq_status', 'political_affilation', 'is_parent', 'religion_important']

    #     df_encoded = pd.get_dummies(result_df, columns=socio_demo_columns, prefix=socio_demo_columns)
    #     df_result = pd.concat([df_encoded,race_dummies], axis=1)

    #     start_col = df_result.columns.get_loc('gender_Female')
    #     end_col = len(df_result.columns)
    #     one_hot_df = df_result.iloc[:,start_col:end_col].astype(int)

    #     return one_hot_df
    
    # def get_one_hot_tensor(self,result_df):

    #     one_hot_df = self.get_one_hot_df(result_df)
    #     return torch.tensor(one_hot_df.to_numpy(),dtype=torch.float32)

