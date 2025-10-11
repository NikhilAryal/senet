# layer wise ReLu sensitivity analysis

'''
We have: a ReLu Budget => calculate layer wise ReLu counts (based on sensitivity)

We then want a layer wise ReLu allocation mask search.

For each layer: binary mask tensor evaluated with activation map. (1 ,0 means presence, absence of ReLu unit)

Layer wise Unit ReLu budget => ReLu allocation mask search
'''
from evaluate import evaluate, eval_subset
# ---Sensitivity analysis---

def compute_relu_sensitivity(model, val_loader, original_accuracy):
    sensitivity_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # Backup original ReLU
            original_relu = module.forward

            # Replace with identity
            module.forward = lambda x: x

            # Evaluate model
            _ ,accuracy = eval_subset(model, val_loader, criterion=torch.nn.CrossEntropyLoss(), device='cpu')
            print(f"Sensitivity of {name}: {original_accuracy - accuracy:.4f}")
            sensitivity_scores[name] = abs(original_accuracy - accuracy)

            # Restore original ReLU
            module.forward = original_relu

    return sensitivity_scores


def allocate_relu_budget(sensitivity_scores, total_relu_budget):
    """
    Allocate ReLU activations across layers based on sensitivity scores.

    Args:
        sensitivity_scores (dict): Layer name -> sensitivity score
        total_relu_budget (int): Total number of ReLU activations allowed

    Returns:
        relu_allocation (dict): Layer name -> number of ReLU activations allocated
    """
    # Normalize sensitivity scores
    total_sensitivity = sum(sensitivity_scores.values())
    normalized_scores = {
        layer: score / total_sensitivity
        for layer, score in sensitivity_scores.items()
    }

    # Allocate ReLU budget proportionally
    relu_allocation = {
        layer: int(normalized_scores[layer] * total_relu_budget)
        for layer in sensitivity_scores
    }

    # Adjust for rounding errors
    allocated = sum(relu_allocation.values())
    while allocated < total_relu_budget:
        # Add remaining ReLUs to the most sensitive layer
        most_sensitive = max(normalized_scores, key=normalized_scores.get)
        relu_allocation[most_sensitive] += 1
        allocated += 1

    return relu_allocation



def relu_allocation(budget, model, proxy_density, num_relu_layers, ):
    n_alpha = get_relu_sensitivity()
    for i in range(num_relu_layers):
        # n_alpha = 
        pass


def get_relu_sensitivity():
    proxy_density = 0.1
    return proxy_density

import torch
import torch.nn as nn
from torchvision.models import resnet18


# Initialize model and dummy input
model = resnet18()
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

for name, module in model.named_modules():
    if isinstance(module, nn.ReLU):
        module.register_forward_hook(get_activation(name))

with torch.no_grad():
    _ = model(input_tensor)

for key, value in activations.items():
    print(f"{key}: {value.shape}")
