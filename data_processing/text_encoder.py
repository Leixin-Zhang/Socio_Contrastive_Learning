import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel



'''
Encoder Classes for converting text to embeddings 

'''


class BERT_CLS:
    def __init__(self, device=None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32):
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)

                cls_emb = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_emb.cpu())

        return torch.cat(all_embeddings, dim=0)



class BERT_MeanPooling:

    def __init__(self, device=None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32):
        all_embeddings = []

        with torch.inference_mode():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)

                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                # mean pooling (optimized)
                mask = attention_mask.unsqueeze(-1)
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)

                mean_pooled = summed / counts
                all_embeddings.append(mean_pooled.cpu())

        return torch.cat(all_embeddings, dim=0)
    


class RoBERTa_MeanPooling:

    def __init__(self, device=None):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32):
        all_embeddings = []

        with torch.inference_mode():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)

                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                mask = attention_mask.unsqueeze(-1)
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)

                mean_pooled = summed / counts
                all_embeddings.append(mean_pooled.cpu())

        return torch.cat(all_embeddings, dim=0)

