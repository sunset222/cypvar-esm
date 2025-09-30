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

        return out_variant, out_drug, attn_v_to_d, attn_d_to_v


class Attention(nn.Module):
 
    def __init__(self, hidden_size, num_heads = 1, drop_out = 0.1):
        super(Attention, self).__init__()
        
        self.norm = nn.LayerNorm(hidden_size)
        self.cross_att = BidirectionalCrossAttention(embed_dim = hidden_size, num_heads = num_heads, dropout_p=drop_out)
        # self.norm2 = nn.LayerNorm(hidden_size)
        # self.pFF = PositionWiseFeedForward(d_model_in = hidden_size, d_model_out = hidden_size * 2, dropout_p=drop_out)
        

    def forward(self, variant, drug, mask_variant, mask_drug , return_attention=False):
        if return_attention:
            drug_att_sum, varaint_att_sum, attn_v_to_d, attn_d_to_v = self.cross_att(variant = self.norm(variant), drug = self.norm(drug), mask_variant = mask_variant, mask_drug=mask_drug)
        #x = x + self.pFF(x = self.norm2(x))
            return variant + drug_att_sum, drug + varaint_att_sum, attn_v_to_d, attn_d_to_v
        else:
            drug_att_sum, varaint_att_sum, _, _ = self.cross_att(variant = self.norm(variant), drug = self.norm(drug), mask_variant = mask_variant, mask_drug=mask_drug)
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

       
   def forward(self, allele, substr, allele_mask, substr_mask, return_attention=False):
       allele_proj = self.allele_w2(self.dropout(self.act(self.allele_w1(allele))))
       substr_proj = self.substr_w2(self.dropout(self.act(self.substr_w1(substr))))
        
       if return_attention:
           allele_out, substr_out, allele_to_substr_att, substr_to_allele_att = self.crossatt(
                    allele_proj,  
                    substr_proj,
                    allele_mask,
                    substr_mask, return_attention=return_attention
           )
           # substr_out = self.allele2substr(
           #          substr_proj,  
           #          allele_proj,   
           #          allele_mask
                    
           # )
       else:
           allele_out, substr_out = self.crossatt(
                   allele_proj,  
                   substr_proj,
                   allele_mask,
                   substr_mask, return_attention=return_attention  
           )

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



       if return_attention:
           return output, allele_to_substr_att, substr_to_allele_att, None
       return output, None, None, None
       



# Attention 추출 사용 예시
def extract_attention_weights(model, allele, substr, allele_mask, substr_mask):
    """
    모델에서 attention weight를 추출하는 함수
    
    Returns:
        - output: 모델 예측값
        - allele_to_substr_att: allele이 substr에 대한 attention (batch_size, num_heads, allele_len, substr_len)
        - substr_to_allele_att: substr이 allele에 대한 attention (batch_size, num_heads, substr_len, allele_len)
    """
    model.eval()
    with torch.no_grad():
        output, allele_to_substr_att, substr_to_allele_att, _ = model(
            allele, substr, allele_mask, substr_mask, return_attention=True
        )
    
    return output, allele_to_substr_att, substr_to_allele_att


# Attention 시각화를 위한 유틸리티 함수
def average_attention_heads(attention_weights):
    """
    여러 head의 attention을 평균내어 하나의 attention map으로 만드는 함수
    
    Args:
        attention_weights: (batch_size, num_heads, query_len, key_len)
    
    Returns:
        averaged_attention: (batch_size, query_len, key_len)
    """
    return attention_weights.mean(dim=1)


def get_top_attended_positions(attention_weights, top_k=5):
    """
    가장 높은 attention을 받는 position들을 찾는 함수
    
    Args:
        attention_weights: (batch_size, query_len, key_len) or (query_len, key_len)
        top_k: 상위 몇 개의 position을 반환할지
    
    Returns:
        top_positions: attention이 높은 (query_idx, key_idx) 튜플들의 리스트
    """
    if len(attention_weights.shape) == 3:
        attention_weights = attention_weights[0]  # 첫 번째 배치만 사용
    
    # Flatten하여 top-k 찾기
    flat_attention = attention_weights.flatten()
    top_k_indices = torch.topk(flat_attention, top_k).indices
    
    # 2D 좌표로 변환
    query_len, key_len = attention_weights.shape
    top_positions = []
    for idx in top_k_indices:
        query_idx = idx // key_len
        key_idx = idx % key_len
        attention_value = attention_weights[query_idx, key_idx].item()
        top_positions.append((query_idx.item(), key_idx.item(), attention_value))
    
    return top_positions