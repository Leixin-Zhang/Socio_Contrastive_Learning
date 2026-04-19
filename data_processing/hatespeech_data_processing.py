import datasets
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from data_processing.text_encoder import BERT_CLS, BERT_MeanPooling, RoBERTa_MeanPooling


class HateSpeechDatasetLoader:
    
    def load_and_preprocess_data(self):
        
        df = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')['train'].to_pandas()

        # data filter (filter annotator who annotated less than 20, filter comment annotated by less than 2 annotators)

        df = df.groupby('annotator_id').filter(lambda x: len(x) >= 20)
        df = df.groupby('comment_id').filter(lambda x: len(x) >= 2)
        df = df.fillna(0)

        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)

        # create 0 / 1 as binary labels (non-hate = 0, hate = 1 )
        df['binary_hatespeech'] = df['hatespeech'].replace(2, 1)
        
        return df


    def get_pivot_df(self, hate_df):

        """
        Build a pivot table for multi-task learning where:
        - each row corresponds to a comment (comment_id)
        - each column corresponds to an annotator (annotator_id)
        - values are binary hate speech labels provided by annotators

        This structure allows modeling each annotator as a separate prediction head.
        """

        pivot = hate_df.pivot(
            index='comment_id',
            columns='annotator_id',
            values='binary_hatespeech'
        )

        return pivot
    


class HateSpeechFeatureBuilder:
    """
    Class for:
    - Text embeddings
    - Socio-demographic embeddings
    - Tensor Construction
    """

    def __init__(self, encoder="SBERT"):

        if encoder == "SBERT":
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
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
        texts = unique_df.text.tolist()

        embeddings = self.encoder.encode(texts)

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
            df_sub['binary_hatespeech'].values,
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
            df_sub.groupby("comment_id")["hatespeech"]
            .mean()
            .gt(0.5)
            .astype(int)
            .reset_index(name="aggregated_hatespeech")
        )


        text_tensor = torch.stack([
            torch.as_tensor(embedding_dict[cid], dtype=torch.float32) 
            for cid in aggregated.comment_id.values])
        
        target_tensor = torch.tensor(
            aggregated["aggregated_hatespeech"].values,
            dtype=torch.float32
        )

        return text_tensor, target_tensor

    # -------------------------------------
    # SOCIO-DEMOGRAPHIC ONE-HOT EMBEDDING
    # -------------------------------------

    def get_one_hot_tensor(self, hate_df, selected_comment_id):

        hate_df = hate_df[hate_df.comment_id.isin(selected_comment_id)]
        start_idx = hate_df.columns.get_loc('annotator_gender_men')
        end_idx = hate_df.columns.get_loc('annotator_sexuality_other')
        one_hot_df = hate_df.iloc[:, start_idx: end_idx+1]

        return torch.tensor(one_hot_df.to_numpy(), dtype=torch.float32)
    
    # -----------------------------------------
    # SOCIO-DEMOGRAPHIC EMBEDDING FROM ENCODERS
    # -----------------------------------------

    def build_socio_embedding_tensor(self, df):



        annotator_embedding_dict = self.annotator_embedding_dict(df)

        annotator_embeddings = torch.stack([
            torch.as_tensor(annotator_embedding_dict[aid], dtype=torch.float32)
            for aid in df.annotator_id.values
        ])

        return annotator_embeddings
        
    
    def annotator_embedding_dict(self, df):

        annotator_df = df.drop_duplicates("annotator_id").copy()

        gender = (
            "gender: " + annotator_df["annotator_gender"].astype(str) +
            annotator_df["annotator_transgender"]
                .map({1: ", transgender", 0: ""})
                .fillna("")
        )

        age = annotator_df["annotator_age"].apply(
            lambda x: f"{int(x)} years old" if pd.notna(x) else "age: unknown"
        )

        education = "education: " + annotator_df["annotator_educ"].fillna("unknown").astype(str).str.replace("_", " ")
        income = "income: " + annotator_df["annotator_income"].fillna("unknown").astype(str).str.replace("_", " ")
        ideology = "ideology: " + annotator_df["annotator_ideology"].fillna("unknown").astype(str).str.replace("_", " ")

        def decode_multi_hot(prefix, name):
            cols = [c for c in df.columns if c.startswith(prefix)]
            values = annotator_df[cols].values

            out = []
            for row in values:
                selected = [
                    cols[i].replace(prefix, "")
                    for i, v in enumerate(row)
                    if str(v) == "1"
                ]
                out.append(f"{name}: " + ", ".join(selected) if selected else f"{name}")
            return out

        religion = decode_multi_hot("annotator_religion_", "religion")
        race = decode_multi_hot("annotator_race_", "race")
        sexuality = decode_multi_hot("annotator_sexuality_", "sexuality")

        texts = [
            f"{g}, {a}, {e}, {i}, {ide}, {r}, {ra}, {s}"
            for g, a, e, i, ide, r, ra, s in zip(
                gender,
                age,
                education,
                income,
                ideology,
                religion,
                race,
                sexuality
            )
        ]
    
        # encode socio-demographic texts to embedding formats
        embeddings = self.encoder.encode(texts)


        return dict(zip(annotator_df.annotator_id.values, embeddings))
    
