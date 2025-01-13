import torch 
import torch.nn as nn 
import torch.optim as optim

class Expert(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(Expert, self).__init__() 
        self.desen_layer = nn.Sequential(
            nn.Linear(input_dim,input_dim),
            nn.Dropout(),
            nn.Linear(input_dim,output_dim),
            nn.Dropout()
        )

    def forward(self, x): 
        # x = torch.relu(self.layer1(x)) 
        x = self.desen_layer(x)
        return x

class Gating(nn.Module): 
    def __init__(self, input_dim, 
            num_experts, dropout_rate=0.1): 
        super(Gating, self).__init__() 
        self.layer1 = nn.Linear(input_dim, num_experts) 

    def forward(self, x): 
        return torch.softmax(self.layer1(x), dim=1)


class MoE(nn.Module): 
    def __init__(self, trained_experts_num=10, input_dim=512, hidden_dim=512, output_dim=512): 
        super(MoE, self).__init__() 
        trained_experts = []
        for expert in range(trained_experts_num):
            trained_experts.append(Expert(input_dim, hidden_dim, output_dim))
        self.experts = nn.ModuleList(trained_experts) 
        num_experts = trained_experts_num

    def forward(self, x, moe_weight): 
        outputs = torch.stack( 
            [expert(x) for expert in self.experts], dim=2) 
        weights = moe_weight.unsqueeze(1).expand_as(outputs) 
        return torch.sum(outputs * weights, dim=2)


class SparseMoE(nn.Module):
    def __init__(self, trained_experts_num=10, input_dim=512, hidden_dim=512, output_dim=512): 
        super(SparseMoE, self).__init__() 
        trained_experts = []
        for expert in range(trained_experts_num):
            trained_experts.append(Expert(input_dim, hidden_dim, output_dim))
        self.experts = nn.ModuleList(trained_experts) 
        num_experts = trained_experts_num
        self.top_k = 1
        self.output_dim = output_dim
    def forward(self, x, moe_weight):
        top_k_logits, indices = moe_weight.topk(self.top_k, dim=-1)
        zeros = torch.full_like(moe_weight, float('-inf'))
        gating_output = zeros.scatter(-1, indices, top_k_logits)
        
        final_output = torch.zeros(x.shape[0], self.output_dim).cuda()

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1).cpu()
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = (expert_output * gating_scores)

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output
