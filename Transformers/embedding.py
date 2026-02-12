import math 
import torch 
import torch.nn as nn 

class EmbeddingWithProjection(nn.Module): 
    def __init__(self,vocab_size,d_embed,d_model,
                 max_position_embedding = 512, dropout = 0.1):
        super().__init__()
        # model's hidden dimension used throughout the transformer layers
        self.d_model = d_model
        # embedding dimension of the input tokens
        self.d_embed = d_embed
        # Number of unique tokens in the vocabulary
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_embed)
        self.projection = nn.Linear(d_embed,d_model)
        self.scaling = float(math.sqrt(d_model))

        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    
    @staticmethod
    def create_positional_encodings(seq_length, d_model, batch_size=1):
        # Creating positional indices [seq_length, 1]
        position = torch.arange(seq_length).unsqueeze(1).float()
        return position


def main():
    vocab_size = 10000
    d_embed = 512
    d_model = 512
    max_position_embedding = 512
    dropout = 0.1

    embedding_layer = EmbeddingWithProjection(vocab_size, d_embed, d_model, max_position_embedding, dropout)
    print(embedding_layer.create_positional_encodings(seq_length=10, d_model=d_model))


if __name__ == "__main__":
    main()