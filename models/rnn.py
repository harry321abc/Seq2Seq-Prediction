import torch
import torch.nn as nn
from models.embed import DataEmbedding

class RNNModel(nn.Module):
    def __init__(self, ninput, nhid, nlayers):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.rnn = nn.RNN(ninput, nhid, nlayers,batch_first=True)
        self.nhid = nhid
        self.nlayers = nlayers
        

    def forward(self, input, h0):
        output, (hn) = self.rnn(input, h0)
        output = self.drop(output)
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
        self.attn = attn
        self.output_attention = output_attention
        self.d_model = d_model
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.RNN_net = RNNModel(ninput = enc_in, nhid = d_model, nlayers= 1)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.device = device

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # TCN_out = self.TCN_net(torch.cat((enc_out, dec_out[:, self.label_len:, :]), dim=1).permute(0, 2, 1))
        x = torch.cat((x_enc, x_dec[:, self.label_len:, :]), dim=1)
        h0 = torch.randn(1, x.shape[0], self.d_model).to(self.device)
        RNN_out = self.RNN_net(x,h0)
        dec_out = self.projection(RNN_out)
        attns = None
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]