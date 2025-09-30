import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.esm.modeling_esm import EsmPreTrainedModel
from transformers import EsmModel
import torch.nn.functional as F

    
import math
def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

            
    def forward(self, x, y, mask):
        batch_size, query_len, embed_dim = x.size()
        _, key_len, _ = y.size()


        query = self.q_proj(x).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(y).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(y).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product


        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))


        attention_probs = self.dropout1(F.softmax(attention_scores, dim=-1))


        attention_output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, query_len, head_dim)


        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)
        output = self.dropout2(self.o_proj(attention_output))

        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model_in, d_model_out, dropout_p=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.W1 = nn.Linear(d_model_in, d_model_out)
        self.W2 = nn.Linear(d_model_out, d_model_in)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.act = nn.GELU()
            
    def forward(self, x):
        return self.dropout2(self.W2(self.dropout1(self.act(self.W1(x)))))


class Transformer(nn.Module):

    def __init__(self, hidden_size, num_heads = 8, dropout_att = 0.1, dropout_pff = 0.1):
        super(Transformer, self).__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mha = MultiHeadAttention(embed_dim = hidden_size, num_heads = num_heads, dropout_p=dropout_att)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.pFF = PositionWiseFeedForward(d_model_in = hidden_size, d_model_out = hidden_size * 4, dropout_p=dropout_pff)
        
    def forward(self, x, y, mask):
        x = x + self.mha(x = self.norm1(x), y = self.norm1(y), mask=mask)
        out = x + self.pFF(self.norm2(x))
        return out

class Featurizer(nn.Module):
    def __init__(self, esm_model, input_size, proj_size, dropout_att = 0.1, dropout_pff = 0.1, num_heads = 8):
        super(Featurizer, self).__init__()
        self.esm = esm_model
        
        self.cross_attn = Transformer(hidden_size = proj_size, num_heads = num_heads, dropout_att = dropout_att, dropout_pff = dropout_pff)
        
        self.proj_out = nn.Linear(input_size, proj_size)
        # self.proj_ffn =  nn.Sequential(
        #                     nn.Linear(input_size, input_size * 2),
        #                     nn.GELU(),
        #                     nn.Dropout(dropout_att),
        #                     nn.Linear(input_size * 2, proj_size))
        


        
        self.norm = nn.LayerNorm(proj_size)
        
    def forward(self, seq1_input_ids, seq1_attention_mask, seq2_input_ids, seq2_attention_mask):
        

 
        outputs1 = self.esm(
            seq1_input_ids,
            attention_mask=seq1_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        outputs2 = self.esm(
            seq2_input_ids,
            attention_mask=seq2_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        sequence_output1, sequence_output2 = outputs1.last_hidden_state, outputs2.last_hidden_state ## WT, VR
        seq1_proj, seq2_proj = self.proj_out(sequence_output1), self.proj_out(sequence_output2)
        out = self.norm(self.cross_attn(seq2_proj, seq1_proj, seq1_attention_mask))
        #out = self.norm(self.cross_attn(sequence_output2, sequence_output1, seq1_attention_mask))

        
        return out
    
class Predictor(nn.Module):

    def __init__(self, hidden_size, num_labels = 1):
        super(Predictor, self).__init__()

        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, num_labels)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.W2(self.act(self.W1(x)))
        return x        
        
class MultiTaskPredictor(nn.Module):
    """
    Multi-task predictor with two prediction heads
    """
    def __init__(self, hidden_size, num_tasks=2):
        super(MultiTaskPredictor, self).__init__()
        
        # Task-specific output layers
        self.task_heads = nn.ModuleList([
            Predictor(hidden_size = hidden_size) for _ in range(num_tasks)
        ])
            
    def mean_valid_tokens(self, batch, attention_mask):
        # batch: (batch_size, seq_len, embed_dim)
        # attention_mask: (batch_size, seq_len)

        attention_mask_expanded = attention_mask.unsqueeze(-1)
        masked_batch = batch * attention_mask_expanded
        mean_batch = masked_batch.sum(dim=1) / attention_mask_expanded.sum(dim=1) # (batch_size, embed_dim)
        return mean_batch

    def forward(self, input_feat, mask):
        # Get pooled representation from valid tokens
        x = self.mean_valid_tokens(input_feat, mask)
        
        # Apply task-specific heads
        outputs = [task_head(x) for task_head in self.task_heads]
        
        return outputs


                
class CYPVarAM(nn.Module):
    def __init__(self, esm_model, num_heads=6, input_size=1280, hidden_size=300, 
                 drop_att=0.1, drop_pff=0.1, num_tasks=2):
        super(CYPVarAM, self).__init__()
        
        self.am_feature = Featurizer(esm_model, input_size=input_size, proj_size=hidden_size, 
                                    dropout_att=drop_att, dropout_pff=drop_pff, num_heads=num_heads)
        self.am_predict = MultiTaskPredictor(hidden_size=hidden_size, num_tasks=num_tasks)
 
        self.num_tasks = num_tasks
        

    def forward(
        self, 
        seq1_input_ids, seq1_attention_mask, seq2_input_ids, seq2_attention_mask, labels=None):
        """
        Forward pass with multi-task outputs
        
        Args:
            labels: Tensor of shape (batch_size, num_tasks) containing the labels for all tasks
        """
        feats = self.am_feature(seq1_input_ids, seq1_attention_mask, seq2_input_ids, seq2_attention_mask)
        logits_list = self.am_predict(feats, seq2_attention_mask)
        
        loss = None
        if labels is not None:
            # Ensure labels is on the correct device
            labels = labels.to(feats.device)
            
            # Choose loss function
            loss_fct_BCE = BCEWithLogitsLoss()
            loss_fct_MSE = MSELoss()
                
            # Calculate loss for each task and sum them
            task_losses = []
            for task_idx in range(self.num_tasks):
                if task_idx == 0:
                    task_logits = logits_list[task_idx].squeeze()
                    task_labels = labels[:, task_idx].squeeze()
                    task_loss = loss_fct_BCE(task_logits, task_labels)
                    task_losses.append(task_loss)
                else:
                    task_logits = logits_list[task_idx].squeeze()
                    task_labels = labels[:, task_idx].squeeze()
                    task_loss = loss_fct_MSE(task_logits, task_labels)
                    task_losses.append(task_loss)
                
            # Sum all task losses
            loss = sum(task_losses) / self.num_tasks
        
        # Stack logits for return
        stacked_logits = torch.cat([logit for logit in logits_list], dim=1)

        return SequenceClassifierOutput(
            loss=loss,
            logits=stacked_logits  # Shape: (batch_size, num_tasks)
        )
    
    def forward_feat(
        self, 
        seq1_input_ids, seq1_attention_mask, seq2_input_ids, seq2_attention_mask):
        
        feats = self.am_feature(seq1_input_ids, seq1_attention_mask, seq2_input_ids, seq2_attention_mask)
        return feats