import torch
import torch.nn as nn
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Expert class
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Define the Gating Network class
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=2)

# Define the Mixture of Experts Layer class
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts,is_print):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
        self.is_print = is_print

    def forward(self, x, num_experts_per_tok):
        gating_scores = self.gate(x)
        topk_gating_scores, topk_indices = gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        if self.is_print:
            print(topk_indices)
        
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=2)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)
        return output

# Define the overall Transformer model with integrated MoE
class TransformerWithMoE(nn.Module):
    def __init__(self, num_layers, dim, head_dim, hidden_dim, n_heads, num_experts, vocab_size, num_experts_per_tok):
        super(TransformerWithMoE, self).__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads) for _ in range(num_layers)])
        self.moe_layer = MoELayer(dim, hidden_dim, dim, num_experts)
        self.output_layer = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.moe_layer(x, self.num_experts_per_tok)
        logits = self.output_layer(x)
        return logits


# Define the overall Transformer model with integrated MoE
class MLPWithMoE(nn.Module):
    def __init__(self, num_layers, dim, hidden_dim, num_experts, num_experts_per_tok,is_print):
        super(MLPWithMoE, self).__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.moe_layer = MoELayer(dim, hidden_dim, hidden_dim, num_experts,is_print)
        self.num_layers = num_layers
    def forward(self, x):
        #print("inital",x.size())
        if self.num_layers>0:
            for layer in self.layers:
                x = layer(x)
        x = self.moe_layer(x, self.num_experts_per_tok)
        #print("after",x.size())
        return x

"""# Initialize the model with configurations matching Mixtral 8x7B
model = MLPWithMoE(
    num_layers=0,              # Number of transformer layers
    dim=4096,                   # Dimension of the model
    hidden_dim=14336,           # Hidden dimensionality in the feed-forward network within the transformer
    num_experts=8,              # Number of experts in the MoE layer
    num_experts_per_tok=2       # Number of experts activated per token
)"""



class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    num_expert = getattr(config,"num_expert",8)
    moe_print = getattr(config,"moe_print",False)
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == 'moe':
        return MLPWithMoE(num_layers=0,dim=config.mm_hidden_size,hidden_dim=config.hidden_size,num_experts=num_expert,num_experts_per_tok=2,is_print=moe_print)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
