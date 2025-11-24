
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, BatchSampler, SequentialSampler
from data.hatespeech_data import HateSpeechDatasetLoader 

def simple_dataloader(text_tensor,target_tensor):

    dataset = TensorDataset(text_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last = (len(text_tensor) % 32 == 1))

    return dataloader

def socio_feature_dataloader(annotator_tensor,text_tensor,target_tensor):

    dataset = TensorDataset(annotator_tensor, text_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last = (len(annotator_tensor) % 32 == 1))

    return dataloader


class MultiTaskDataset(Dataset):

    def __init__(self, text_embedding, pivot_df):
        self.text_embedding = text_embedding
        
        filled_df = pivot_df.fillna(-1)
        self.all_targets = torch.tensor(filled_df.values, dtype=torch.float32)
        
        self.all_masks = torch.tensor(
            pivot_df.notna().astype(int).values, 
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.text_embedding)
        
    def __getitem__(self, idx):
        return self.text_embedding[idx], self.all_targets[idx], self.all_masks[idx]
    

def multi_task_dataloader(hate_df, embeddings, train_comment_id, test_comment_id,comment_mapping_dict):
    
    pivot = hate_df.pivot(index='comment_id', columns='annotator_id', values='binary_hatespeech')
    train_unique_tensor = torch.tensor(
        np.array([embeddings[comment_mapping_dict[i]] for i in train_comment_id]), 
        dtype=torch.float32)
    test_unique_tensor = torch.tensor(
        np.array([embeddings[comment_mapping_dict[i]] for i in test_comment_id]), 
        dtype=torch.float32)
        
    train_dataset = MultiTaskDataset(train_unique_tensor, pivot)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = MultiTaskDataset(test_unique_tensor, pivot)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader,test_dataloader



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

def contrastive_dataloader(train_socio,train_text_tensor,train_target,train_comment_id,
                          test_socio,test_text_tensor,test_target,test_comment_id):

    train_dataset = ContrastiveDataset(train_socio, train_text_tensor, train_target, train_comment_id)
    batch_sampler = CommentIDBatchSampler(
        train_dataset.comment_id,
        batch_size=512,
        shuffle=(len(train_text_tensor)%512 ==1)
    )
    contrasive_train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    test_dataset = TensorDataset(test_socio, test_text_tensor, test_target)
    contrasive_test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return contrasive_train_dataloader, contrasive_test_dataloader


## comment id sampler for contrasive learning, try to put text with the comment_id into the same batch 

class CommentIDBatchSampler:
    def __init__(self, comment_ids, batch_size, shuffle=True):
        self.comment_ids = comment_ids
        self.batch_size = batch_size
        self.shuffle = shuffle

        # group in terms of comment_ids
        self.id_to_indices = {}
        for idx, cid in enumerate(comment_ids):
            if cid not in self.id_to_indices:
                self.id_to_indices[cid] = []
            self.id_to_indices[cid].append(idx)

        # count of each comment_ids
        self.comment_lengths = {cid: len(indices) for cid, indices in self.id_to_indices.items()}

    def __iter__(self):
        # shuffle comment_ids
        comment_ids = list(self.id_to_indices.keys())
        if self.shuffle:
            np.random.shuffle(comment_ids)

        batch = []
        for cid in comment_ids:
            indices = self.id_to_indices[cid]
            if self.shuffle:
                np.random.shuffle(indices)

            # put the text with the same comment_id into the same batch
            batch.extend(indices)

            # yield if exceeding the batch size
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        # deal with remaining data
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.comment_ids) + self.batch_size - 1) // self.batch_size


