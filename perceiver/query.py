import torch
import torch.nn as nn


class Query_Gen(nn.Module):
    def __init__(self, input_dim, batch_size, dim, num_filters=128, hidden_dim=512, dropout=float(0.0)):
        super(Query_Gen, self).__init__()

        self.dim = dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # 1D CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2)
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(self.num_filters, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.dim * self.dim * 2 // self.batch_size),
            nn.Dropout(dropout)
        )

        

    def forward(self, x):
        
        
        
        x = x.permute(0, 2, 1)  # Convert (batch_size, input_dim, sequence_length) to (batch_size, sequence_length, input_dim)
        x = self.cnn_layers(x)
        x = x.squeeze(0).permute(1,0)
        
        
        output = self.mlp_layers(x)
        # Reshape output to desired shape (batch_size, dim, dim)
        output = output.view(1, self.dim, self.dim)
        
        return output

'''
input_dim = 4
dim = 1000
batch_size = 200
model = Query_Gen(input_dim, batch_size, dim)

input_data = torch.rand(1, batch_size, input_dim)

output = model(input_data)
print(output.shape)  
'''