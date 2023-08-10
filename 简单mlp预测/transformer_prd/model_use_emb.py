import torch
import torch.nn as nn

# Define the input and hidden dimensions
input_dim = 1000
hidden_dim = 256
# Create an instance of the embedding layer
embedding = nn.Embedding(input_dim, hidden_dim)
# Generate some random input indices
batch_size = 32
seq_length = 10
matrix = torch.randn(batch_size, seq_length)

input_indices = torch.randint(input_dim, (batch_size, seq_length))

print(matrix.shape, input_indices.shape)

# Pass the input indices through the embedding layer
embedded = embedding(matrix.long())
# Print the shape of the embedded output
print(embedded.shape)
