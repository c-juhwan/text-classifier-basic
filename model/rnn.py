import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, rnn_layer_num, dropout_ratio=0.1):
        super(ClassifierRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.output_size = embed_size
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        nn.init.uniform_(self.embed.weight, -0.25, 0.25)

        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=embed_size, num_layers=rnn_layer_num, batch_first=True, dropout=dropout_ratio)
        self.linear_out = nn.Linear(in_features=self.output_size, out_features=num_classes)

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.activation = nn.ReLU()
    
    def forward(self, input_ids):
        # embed the input_ids into embedding vectors & dropout
        input_tensor = self.embed(input_ids)
        input_tensor = self.dropout(input_tensor)

        # pack the input_tensor into a PackedSequence
        non_pad_len = input_ids.ne(0).sum(dim=1).cpu()
        packed_input_tensor = nn.utils.rnn.pack_padded_sequence(input_tensor, non_pad_len, batch_first=True, enforce_sorted=False)

        # apply the LSTM
        output, (hidden, cell) = self.rnn(packed_input_tensor)
        hidden = hidden.squeeze() # requires layer_num == 1
        
        # apply the output layer
        output = self.dropout(hidden)
        output = self.linear_out(output)

        return output