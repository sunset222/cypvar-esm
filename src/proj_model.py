

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.varmodel_MT_AllLoRA import gelu, MultiHeadAttention, Transformer
import math

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Shared Q, K projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        # Separate V and O projections
        self.v_proj_variant = nn.Linear(embed_dim, embed_dim)
        #self.o_proj_variant = nn.Linear(embed_dim, embed_dim)

        self.v_proj_drug = nn.Linear(embed_dim, embed_dim)
        #self.o_proj_drug = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, variant, drug, mask_variant=None, mask_drug=None):
        B, Lv, D = variant.shape
        _, Ld, _ = drug.shape

        # Project Q and K
        q = self.q_proj(variant).view(B, Lv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lv, D]
        k = self.k_proj(drug).view(B, Ld, self.num_heads, self.head_dim).transpose(1, 2)     # [B, H, Ld, D]

        # Shared score matrix
        raw_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, Lv, Ld]


        
        # === Variant → Drug ===
        score_v_to_d = raw_score.clone()
        if mask_drug is not None:
            score_v_to_d = score_v_to_d.masked_fill(mask_drug[:, None, None, :] == 0, float('-inf'))

        attn_v_to_d = F.softmax(score_v_to_d, dim=-1)
        attn_v_to_d = attn_v_to_d

        v_drug = self.v_proj_drug(drug).view(B, Ld, self.num_heads, self.head_dim).transpose(1, 2)
        out_variant = torch.matmul(attn_v_to_d, v_drug).transpose(1, 2).contiguous().view(B, Lv, D)
        out_variant = self.dropout(out_variant)

        

        # === Drug → Variant ===
        score_d_to_v = raw_score.transpose(-2, -1).clone()
        if mask_variant is not None:
            score_d_to_v = score_d_to_v.masked_fill(mask_variant[:, None, None, :] == 0, float('-inf'))

        attn_d_to_v = F.softmax(score_d_to_v, dim=-1)
        attn_d_to_v = attn_d_to_v

        v_variant = self.v_proj_variant(variant).view(B, Lv, self.num_heads, self.head_dim).transpose(1, 2)
        out_drug = torch.matmul(attn_d_to_v, v_variant).transpose(1, 2).contiguous().view(B, Ld, D)
        out_drug = self.dropout(out_drug)

        return out_variant, out_drug
        
# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout_p=0.1):
#         super(MultiHeadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."
        
        
#         #self.out_norm = nn.LayerNorm(embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.o_proj = nn.Linear(embed_dim, embed_dim)
        
#         self.drop_out = nn.Dropout(dropout_p)
            
#     def forward(self, x, y, mask):
#         batch_size, query_len, embed_dim = x.size()
#         _, key_len, _ = y.size()


#         query = self.q_proj(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key = self.k_proj(y).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
#         value = self.v_proj(y).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        
#         # query = x.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
#         # key = y.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
#         # value = y.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product


#         if mask is not None:
#             attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))


#         attention_probs = F.softmax(attention_scores, dim= -1)


#         attention_output = torch.matmul(attention_probs, value) # (batch_size, num_heads, query_len, head_dim)


#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)
#         #output = self.drop_out(attention_output)
#         output = self.drop_out(self.o_proj(attention_output))

#         return output
    
# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_model_in, d_model_out, dropout_p=0.1):
#         super(PositionWiseFeedForward, self).__init__()
#         self.W1 = nn.Linear(d_model_in, d_model_out)
#         self.W2 = nn.Linear(d_model_out, d_model_in)
#         self.dropout1 = nn.Dropout(dropout_p)
#         self.dropout2 = nn.Dropout(dropout_p)
#         self.act = nn.GELU()
            
#     def forward(self, x):
#         return self.dropout2(self.W2(self.dropout1(self.act(self.W1(x)))))


class Attention(nn.Module):
 
    def __init__(self, hidden_size, num_heads = 1, drop_out = 0.1):
        super(Attention, self).__init__()
        
        self.norm = nn.LayerNorm(hidden_size)
        self.cross_att = BidirectionalCrossAttention(embed_dim = hidden_size, num_heads = num_heads, dropout_p=drop_out)
        # self.norm2 = nn.LayerNorm(hidden_size)
        # self.pFF = PositionWiseFeedForward(d_model_in = hidden_size, d_model_out = hidden_size * 2, dropout_p=drop_out)
        

    def forward(self, variant, drug, mask_variant, mask_drug):
        drug_att_sum, varaint_att_sum = self.cross_att(variant = self.norm(variant), drug = self.norm(drug), mask_variant = mask_variant, mask_drug=mask_drug)
        #x = x + self.pFF(x = self.norm2(x))
        return variant + drug_att_sum, drug + varaint_att_sum
        
class InteractionModel(nn.Module):
   def __init__(self, mol_input_dim, prot_input_dim, hidden_dim=128, 
                 num_heads=1, drop_out=0.1):
        
       super().__init__()

       self.act = nn.GELU()


       self.allele_w1 = nn.Linear(prot_input_dim, hidden_dim)
       self.substr_w1 = nn.Linear(mol_input_dim, hidden_dim)
       self.allele_w2 = nn.Linear(hidden_dim, hidden_dim)
       self.substr_w2 = nn.Linear(hidden_dim, hidden_dim)



       self.crossatt = Attention(hidden_size=hidden_dim, num_heads=num_heads, drop_out=drop_out)
       #self.substr2allele = Attention(hidden_size=hidden_dim, num_heads=num_heads, drop_out=drop_out)

    
       self.regression_w1 = nn.Linear(hidden_dim, 1)
       #self.regression_w2 = nn.Linear(hidden_dim // 4, 1)


       self.dropout = nn.Dropout(drop_out)
       #self.norm = nn.LayerNorm(hidden_dim)

       
   def forward(self, allele, substr, allele_mask, substr_mask):
       allele_proj = self.allele_w2(self.dropout(self.act(self.allele_w1(allele))))
       substr_proj = self.substr_w2(self.dropout(self.act(self.substr_w1(substr))))
        

       allele_out, substr_out = self.crossatt(
                allele_proj,  
                substr_proj,
                allele_mask,
                substr_mask  
       )
       # substr_out = self.allele2substr(
       #          substr_proj,  
       #          allele_proj,   
       #          allele_mask
                
       # )

       allele_mask = allele_mask.float().unsqueeze(-1)
       substr_mask = substr_mask.float().unsqueeze(-1)

       allele_att_pooled = (allele_out * allele_mask).sum(1) / allele_mask.sum(1)
       substr_att_pooled = (substr_out * substr_mask).sum(1) / substr_mask.sum(1)




       combined = 0.5 * (allele_att_pooled + substr_att_pooled)
       #self.norm(
       
       
       #torch.cat([allele_att_pooled, substr_att_pooled], dim=1)#self.norm(0.5 * (allele_out[:, 0, :] + substr_out[:, 0, :]))
       #0.5 * (allele_out[:, 0, :] + substr_out[:, 0, :])
       #torch.cat([allele_att_pooled, substr_att_pooled], dim=1)#allele_att_pooled + substr_att_pooled

       #output = self.act(self.regression_w1(combined))
       output = self.regression_w1(combined)



       
       return output, None, None, None




   

   



   

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.varmodel_MT_AllLoRA import gelu, MultiHeadAttention, Transformer
# import math

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout_p=0.1):
#         super(MultiHeadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."

#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        
#         self.drop_out = nn.Dropout(dropout_p)
            
#     def forward(self, x, y, mask):
#         batch_size, query_len, embed_dim = x.size()
#         _, key_len, _ = y.size()


#         query = self.q_proj(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key = self.k_proj(y).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
#         value = self.v_proj(y).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        
#         # query = x.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
#         # key = y.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
#         # value = y.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product


#         if mask is not None:
#             attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))


#         attention_probs = F.softmax(attention_scores, dim=-1)


#         attention_output = torch.matmul(attention_probs, value) # (batch_size, num_heads, query_len, head_dim)


#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)
#         #output = self.drop_out(attention_output)
#         output = self.drop_out(self.o_proj(attention_output))

#         return output
    

# class Attention(nn.Module):
 
#     def __init__(self, hidden_size, num_heads = 1, drop_out = 0.1):
#         super(Attention, self).__init__()

#         self.norm = nn.LayerNorm(hidden_size)
#         self.cross_att = MultiHeadAttention(embed_dim = hidden_size, num_heads = num_heads, dropout_p=drop_out)
        

#     def forward(self, x, y, y_mask):
#         x = x + self.cross_att(x = self.norm(x), y = self.norm(y), mask=y_mask)

#         return x
        
# class InteractionModel(nn.Module):
#    def __init__(self, mol_input_dim, prot_input_dim, hidden_dim=128, 
#                  num_heads=1, drop_out=0.1):
        
#        super().__init__()

#        self.act = nn.GELU()


#        self.allele_w1 = nn.Linear(prot_input_dim, hidden_dim)
#        self.substr_w1 = nn.Linear(mol_input_dim, hidden_dim)
#        self.allele_w2 = nn.Linear(hidden_dim, hidden_dim)
#        self.substr_w2 = nn.Linear(hidden_dim, hidden_dim)

#        # self.allele_norm = nn.LayerNorm(hidden_dim)
#        # self.substr_norm = nn.LayerNorm(hidden_dim)
#        self.drop_out = nn.Dropout(drop_out)
    
#        self.allele2substr = Attention(hidden_size=hidden_dim, num_heads=num_heads, drop_out=drop_out)
#        self.substr2allele = Attention(hidden_size=hidden_dim, num_heads=num_heads, drop_out=drop_out)

    
#        self.regression_w1 = nn.Linear(hidden_dim*2, 1)
#        #self.regression_w2 = nn.Linear(hidden_dim // 4, 1)




#        #self.norm = nn.LayerNorm(hidden_dim)


       
#    def forward(self, allele, substr, allele_mask, substr_mask):

#        allele_proj = self.allele_w2(self.drop_out(self.act(self.allele_w1(allele))))
#        substr_proj = self.substr_w2(self.drop_out(self.act(self.substr_w1(substr))))
        

#        allele_out = self.allele2substr(
#                 allele_proj,  
#                 substr_proj,
#                 substr_mask  
#        )
#        substr_out = self.substr2allele(
#                 substr_proj,  
#                 allele_proj,   
#                 allele_mask
                
#        )

#        allele_mask = allele_mask.float().unsqueeze(-1)
#        substr_mask = substr_mask.float().unsqueeze(-1)

#        allele_att_pooled = (allele_out * allele_mask).sum(1) / allele_mask.sum(1)
#        substr_att_pooled = (substr_out * substr_mask).sum(1) / substr_mask.sum(1)




#        combined = torch.cat([allele_att_pooled, substr_att_pooled], dim=1)#allele_att_pooled + substr_att_pooled
#        #self.norm(
       
       
#        #torch.cat([allele_att_pooled, substr_att_pooled], dim=1)#self.norm(0.5 * (allele_out[:, 0, :] + substr_out[:, 0, :]))
#        #0.5 * (allele_out[:, 0, :] + substr_out[:, 0, :])
#        #torch.cat([allele_att_pooled, substr_att_pooled], dim=1)#allele_att_pooled + substr_att_pooled

#        #output = self.act(self.regression_w1(combined))
#        output = self.regression_w1(combined)



       
#        return output, None, None, None

   