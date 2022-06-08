import torch
import torch.nn as nn
from models.embed import DataEmbedding

class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input):
        output = self.fc1(input)
        output = self.relu1(output)
        output = self.fc3(output)
        return output

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.seq_len = seq_len  
        self.label_len = label_len
        self.pred_len = out_len
        self.c_out = c_out
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.ANN_net = ANNModel(input_dim = (self.seq_len +self.pred_len) * enc_in, hidden_dim = d_model, output_dim= (self.seq_len +self.pred_len)* self.c_out)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x = torch.cat((x_enc, x_dec[:, self.label_len:, :]), dim=1)
        x = torch.reshape(x,(x.size()[0],-1))
        ANN_out = self.ANN_net(x)
        dec_out = torch.reshape(ANN_out, (ANN_out.size()[0],-1, self.c_out))
        attns = None
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :] # [B, L, D]