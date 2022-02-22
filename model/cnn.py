import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, dropout_ratio=0.1):
        """Initialize the model by setting up the layers.
        """
        super(ClassifierCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.each_out_size = embed_size // 3
        self.output_size = self.each_out_size * 3
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        nn.init.uniform_(self.embed.weight, -0.25, 0.25)

        self.conv_k3 = nn.Conv1d(in_channels=self.embed_size, out_channels=self.each_out_size, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv_k4 = nn.Conv1d(in_channels=self.embed_size, out_channels=self.each_out_size, kernel_size=4, stride=1, padding=3, bias=False)
        self.conv_k5 = nn.Conv1d(in_channels=self.embed_size, out_channels=self.each_out_size, kernel_size=5, stride=1, padding=4, bias=False)

        self.linear1 = nn.Linear(in_features=self.output_size, out_features=self.output_size)
        self.linear2 = nn.Linear(in_features=self.output_size, out_features=self.output_size)
        self.linear_out = nn.Linear(in_features=self.output_size, out_features=num_classes)

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.activation = nn.ReLU()
    
    def forward(self, input_ids):
        # embed the input_ids into embedding vectors & dropout
        input_tensor = self.embed(input_ids) # (batch_size, seq_len, embed_size)
        input_tensor = self.dropout(input_tensor)
        input_tensor = input_tensor.transpose(1, 2) # (batch_size, embed_size, seq_len)

        # apply each convolution & concatenate the outputs
        conv3 = torch.max(self.conv_k3(input_tensor), dim=2) # (batch_size, each_out_size)
        conv4 = torch.max(self.conv_k4(input_tensor), dim=2)
        conv5 = torch.max(self.conv_k5(input_tensor), dim=2)

        hidden = torch.concat([conv3.values, conv4.values, conv5.values], dim=1) # (batch_size, output_size)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # apply two linear layers
        hidden = self.dropout(hidden)
        hidden = self.activation(self.linear1(hidden))
        hidden = self.dropout(hidden)
        hidden = self.activation(self.linear2(hidden))

        # apply the output layer
        output = self.dropout(hidden)
        output = self.linear_out(output)

        return output