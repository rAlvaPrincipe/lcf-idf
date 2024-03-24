from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch


def cat2id2cat(cats):
    cats = sorted(cats)
    id2cat = dict()
    for cat in cats:
        id2cat[cats.index(cat)] = cat
    cat2id = {v: k for k, v in id2cat.items()}
    return cat2id, id2cat


class TextDataset(Dataset):
    def __init__(self, docs, labels):
        self.docs = docs
        self.integer_labels = labels

    def __len__(self):
        return len(self.integer_labels)

    def __getitem__(self, idx):
        return self.docs[idx], self.integer_labels[idx]


def text_data_loader(docs, vec_labels, batch):
    data = TextDataset(docs, vec_labels)
    return DataLoader(data, batch_size=batch, shuffle=True)


def bert_data_loader(tokenizer, docs, y, batch, n_tokens):
    encodings = tokenizer(docs, truncation=True, padding=True, max_length=n_tokens, add_special_tokens=True, return_tensors='pt')
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask']),
        torch.tensor(y)
    )
    return DataLoader(train_dataset, batch_size=batch, shuffle=True)


class AutoencoderDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]
        