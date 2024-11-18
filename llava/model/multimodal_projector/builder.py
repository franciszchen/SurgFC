import torch
import torch.nn as nn
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F



"""
#Expert module
class Expert(nn.Module):

    def __init__(self, input_dim,out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)




# First define the top k router module 
class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear =nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.linear(mh_output)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices



#Changing the above to accomodate noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class MoE(nn.Module):
    def __init__(self, n_embed, output_dim, num_experts, top_k):
        super(MoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)  #n_embed dim->num_of expert
        self.experts = nn.ModuleList([Expert(n_embed,output_dim) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        #print("MoE load tensor")
        gating_output, indices = self.router(x)
        #print("selected experts: ",indices)
        #final_output = torch.zeros_like(x)
        #final_output = final_output.repeat(1,1,4)

        a, b, c = x.shape

        # 定义新张量的形状为(a, b, 4*c)
        new_shape = (a, b, 4*c)

        # 创建新的张量，它是全零张量，dtype和device与existing_tensor相同
        final_output = torch.zeros(new_shape, dtype=x.dtype, device=x.device)

        # Reshape inputs for batch processing
        # 每一个token走不同的gate，flat_x把batch和token展开到一个维度
        flat_x = x.reshape(-1, x.size(-1))
        # 每一个token走不同的gate，gating_output也把batch和token展开到一个维度
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        #print("entering loop")
        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            
            # Create a mask for the inputs where the current expert is in top-k 比如i=0时，找到要经过expert0的input
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1) #展开

            if flat_mask.any():
                expert_input = flat_x[flat_mask] #找到和expert对应的输入
                expert_output = expert(expert_input) #对应的输入进入expert得到对应的输出

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1) #得到output应该对应的权重
                weighted_output = expert_output * gating_scores #把值和expert相乘

                # Update final output
                # We need to scatter_add the weighted outputs to their original positions in the batch
                final_output.masked_scatter_(expert_mask.unsqueeze(-1), weighted_output)  #把输出放回到之前的位置,scatter用法是更新位置上在mask上为True的值，其余保持不变

        print("final_output:",final_output.shape," ",final_output.device())
        return final_output
        """







class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

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
        # Note that we're now only using input_dim to determine the gating scores
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # We only need to pass the first token's feature from each sequence to the gate
        # if we assume all tokens in the batch should go through the same gate.
        # We use the mean of the features across the token dimension to represent the sequence.
        mean_x = x.mean(dim=1)
        return F.softmax(self.gate(mean_x), dim=1)

# Define the Mixture of Experts Layer class
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_experts_per_tok):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = NoisyTopkRouter(input_dim, num_experts,num_experts_per_tok)
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x):

        gating_scores, topk_indices = self.gate(x)

        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=2)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)
        return output

class MLPWithMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_experts_per_tok):
        super(MLPWithMoE, self).__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer = MoELayer(dim, hidden_dim, hidden_dim, num_experts,num_experts_per_tok)
    def forward(self, x):
        x = self.moe_layer(x)
        #print("after",x.size())
        return x


"""# Initialize the model with configurations matching Mixtral 8x7B
model = MLPWithMoE(
    num_layers=0,              # Number of transformer layers
    dim=4096,                   # Dimension of the model
    hidden_dim=14336,           # Hidden dimensionality in the feed-forward network within the transformer
    num_experts=8,              # Number of experts in the MoE layer
    num_experts_per_tok=2       # Number of experts activated per token
)
"""



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
        return MLPWithMoE(dim=config.mm_hidden_size,hidden_dim=config.hidden_size,num_experts=8,num_experts_per_tok=2)

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
