import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from data_processing.hatespeech_data_processing import HateSpeechDatasetLoader 
from collections import defaultdict

def simple_dataloader(text_tensor,target_tensor,shuffle):
    '''
    Dataloader for loading text, target 
    '''
    dataset = TensorDataset(text_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=32, drop_last = (len(text_tensor) % 32 == 1), shuffle=shuffle)

    return dataloader

def annotator_feature_dataloader(annotator_tensor,text_tensor,target_tensor,shuffle):
    '''
    Dataloader for loading socio-demographic features, text, target 
    '''
    dataset = TensorDataset(annotator_tensor, text_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=32, drop_last = (len(annotator_tensor) % 32 == 1), shuffle=shuffle)

    return dataloader



class MultiTaskDataset(Dataset):
    
  
    def __init__(self, embedding_tensor, target_tensor, mask_tensor):
        
        self.embedding_tensor = embedding_tensor
        self.target_tensor = target_tensor
        self.mask_tensor = mask_tensor
        
        # test if the shape matches
        assert len(embedding_tensor) == len(target_tensor) == len(mask_tensor), \
        "Embedding, target, and mask tensors must have the same length."
        
    def __len__(self):
        return len(self.embedding_tensor)
        
    def __getitem__(self, idx):
        return self.embedding_tensor[idx], self.target_tensor[idx], self.mask_tensor[idx]



def multi_task_dataloader(pivot, embedding_dict, comment_ids, batch_size=32):
    
    """    
    Args:
        pivot_df: DataFrame with unique_text_id as index, 
                    unique_annotator_id as column name, 
                    filled with each annotator's label to each text, 
                    (NA when this annotator doesn't annotator the text/comment)
        embeddings: unique text/comment embeddings
        train comment_ids: text/comment_id list in train dataset
        comment_mapping_dict: comment_id to text embedding mapping 

    """  

    pivot_sub = pivot.loc[comment_ids]

    #  targets（fill na with 0）
    targets = pivot_sub.fillna(0).values

    #  mask places not annotated by specific annotators
    masks = pivot_sub.notna().astype(int).values

    embeddings = np.stack([embedding_dict[cid] for cid in pivot_sub.index])

    emb = torch.tensor(embeddings, dtype=torch.float32)
    tar = torch.tensor(targets, dtype=torch.float32)
    mask = torch.tensor(masks, dtype=torch.float32)

    dataset = MultiTaskDataset(emb, tar, mask)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


    
def contrastive_dataloader(train_socio,train_text_tensor,train_target,train_comment_id):
    """
    Args:
        train socio: one_hot socio-demo features 
        train_text_tensor: text embedding tensor
        train_target: target tensor

    """  
    train_dataset = ContrastiveDataset(train_socio, train_text_tensor, train_target, train_comment_id)

    batch_sampler = CommentIDBatchSampler(
        train_dataset.comment_id,
        batch_size=512,
    )
    contrasive_train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    

    return contrasive_train_dataloader



class ContrastiveDataset(Dataset):
    
    def __init__(self, social, text, target, comment_id):
        self.social = social
        self.text = text
        self.target = target
        self.comment_id = comment_id
        
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (
            self.social[idx],
            self.text[idx],
            self.target[idx],
            self.comment_id[idx]
        )
    


class CommentIDBatchSampler:
    """
    Group samples with the same comment_id (from training_data) into batches.
    """

    def __init__(self, comment_ids, batch_size, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle

        # 1️. group indices by comment_id
        self.groups = defaultdict(list)
        for idx, cid in enumerate(comment_ids):
            self.groups[cid].append(idx)

    def __iter__(self):

        # 2️. get all groups
        groups = list(self.groups.values())

        if self.shuffle:
            np.random.shuffle(groups)
            for g in groups:
                np.random.shuffle(g)

        # 3️. flatten
        all_indices = [idx for group in groups for idx in group]

        # 4️. yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]

    def __len__(self):
        return (sum(len(g) for g in self.groups.values()) + self.batch_size - 1) // self.batch_size
    

