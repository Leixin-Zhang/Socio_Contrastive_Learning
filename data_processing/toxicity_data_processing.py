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

