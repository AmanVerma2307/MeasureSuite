import torch
#from utils.summary import print_model_summary

class MHSA(torch.nn.Module):
    
    """
    Multi head self-attention
    """

    def __init__(self,d_in,d_model,num_heads):

        super().__init__()
        self.d_in = d_in # input dimensions
        self.d_model = d_model # model dimensions
        self.num_heads = num_heads # number of heads

        assert d_model % num_heads == 0

        self.depth = self.d_model // num_heads

        self.query_dense = torch.nn.Linear(self.d_in,
                                           self.d_model)
        self.key_dense = torch.nn.Linear(self.d_in,
                                         self.d_model)
        self.value_dense = torch.nn.Linear(self.d_in,
                                           self.d_model)
        
        self.projection_dense = torch.nn.Linear(self.d_model,
                                                self.d_model)
        
    def forward(self, x):

        """
        Multi head self-attention

        INPUTS:-
        1) x: Input tokens of shape [B,N,d_model]

        OUTPUTS:-
        1) z: Output tokens of shape [B,N,d_model]
        """

        B, N, d_in = x.shape # Extracting dimensions

        q = self.query_dense(x) # Query -> [B,N,depth]
        k = self.key_dense(x) # Key -> [B,N,depth]
        v = self.value_dense(x) # Value -> [B,N,depth]

        q = q.view(B,N,self.num_heads,self.depth).transpose(1,2) # q -> [B,H,N,depth]
        k = k.view(B,N,self.num_heads,self.depth).transpose(1,2) # k -> [B,H,N,depth]
        v = v.view(B,N,self.num_heads,self.depth).transpose(1,2) # v -> [B,H,N,depth]

        attn = q@k.transpose(2,3) # QK^T -> [B,H,N,N]
        attn = torch.softmax(attn/k.size(-1)**0.5,dim=-1) # Attention matrix -> [B,H,N,N]

        z = attn@v # Attention summation -> [B,H,N,depth]
        z = z.transpose(1,2).contiguous().view(B,N,self.d_model) # Reshape -> [B,N,d_model]
        z = self.projection_dense(z) # Projection -> [B,N,d_model]
        return z
    
class encoder(torch.nn.Module):

    """
    Encoder module
    """

    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 rate,
                 max_seq_len):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.max_seq_len = max_seq_len

        self.att = MHSA(self.d_model,
                        self.d_model,
                        self.num_heads)
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.d_model,self.dff),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dff,self.d_model),
            torch.nn.ReLU(),
        )

        self.layernorm1 = torch.nn.LayerNorm([self.max_seq_len,self.d_model],eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm([self.max_seq_len,self.d_model],eps=1e-6)

        self.droput1 = torch.nn.Dropout(self.rate)
        self.droput2 = torch.nn.Dropout(self.rate)

    def forward(self,x):

        attn_output = self.att(x) # MHSA
        attn_output = self.droput1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.droput2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

if __name__ == "__main__":

    ip = torch.normal(0,1,(10,288,32))
    #ip_head = torch.normal(0,1,(10,288,32))

    #mhsa = MHSA(32,32,16)
    
    #print(mhsa(ip_head).size())
    #print_model_summary(mhsa,
    #                    (288,32))

    encoder_layer = encoder(512,
                      16,
                      128,
                      0.3,
                      288)
    
    #op = encoder_layer(ip)
    #print(op.size())

    total_params = sum(p.numel() for p in encoder_layer.parameters())
    print('Total parameters: '+str(total_params))

    




