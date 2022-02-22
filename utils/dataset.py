import os
import pandas as pd

import torch
import torchtext
from torchtext.data.utils import get_tokenizer

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, text_list, label_list, vocab:torchtext.vocab.Vocab):
        self.text_list = text_list
        self.label_list = label_list
        self.args = args
        self.vocab = vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.datas = []
        
        if self.args.max_seq_len is None:
            args.max_seq_len = max(len(t.split()) for t in text_list)

        # Tokenize text
        for each_text, each_label in zip(text_list, label_list):
            if len(each_text.split()) > args.max_seq_len - 2: # -2 for <sos> and <eos>
                truncated_text = each_text[:args.max_seq_len - 2]
            else:
                truncated_text = each_text

            text_encoded = [] 
            text_encoded.append(self.vocab['<sos>'])
            for token in self.tokenizer(truncated_text.lower()):
                text_encoded.append(self.vocab[token])
            text_encoded.append(self.vocab['<eos>'])

            # Pad text
            if len(text_encoded) < args.max_seq_len:
                text_encoded.extend([self.vocab['<pad>']] * (args.max_seq_len - len(text_encoded)))
            
            data = {}
            data['text'] = torch.Tensor(text_encoded).long()
            data['label'] = torch.Tensor([each_label]).long()
            self.datas.append(data)
        
    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)

def build_dataset(args, vocab:torchtext.vocab.Vocab):
    df = {}
    df['train'] = pd.read_csv(args.train_data_path, header=None)
    df['test'] = pd.read_csv(args.test_data_path, header=None)

    valid_split_index = int(len(df['train']) * (1 - args.valid_split_ratio))
    df['valid'] = df['train'].iloc[valid_split_index:]
    df['train'] = df['train'].iloc[:valid_split_index]

    text_column_index = args.text_column_index
    label_column_index = args.label_column_index

    datasets = {}
    for each in ['train', 'valid', 'test']:
        text_list = df[each][text_column_index]
        label_list = df[each][label_column_index]

        datasets[each] = CustomDataset(args, text_list, label_list, vocab)

    return datasets