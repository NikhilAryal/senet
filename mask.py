import torch
import torch.nn as nn


def initialize_relu_mask(activation_shape, relu_budget):
    # mask = torch.zeros(activation_shape, dtype=torch.uint8)
    # indices = torch.randperm(mask.numel())[:relu_budget]
    # mask.view(-1)[indices] = 1
    # return mask

    total_positions = torch.prod(torch.tensor(activation_shape))
    mask = torch.zeros(total_positions, dtype=torch.uint8)
    indices = torch.randperm(total_positions)[:relu_budget]
    mask[indices] = 1
    return mask.view(activation_shape)


# update mask tensor as Senet is trained
def update_relu_mask(pr_activations, ar_activations, relu_budget):
    diff = (pr_activations - ar_activations).abs()
    flat_diff = diff.view(-1)
    top_indices = torch.topk(flat_diff, relu_budget).indices
    new_mask = torch.zeros_like(flat_diff, dtype=torch.uint8)
    new_mask[top_indices] = 1
    return new_mask.view_as(diff)


def update_relu_mask_nodict(pr_activations, ar_activations, relu_budget):
    updated_masks = {}

    for layer_name in pr_activations:
        pr = pr_activations[layer_name]
        ar = ar_activations[layer_name]

        # Compute absolute difference
        diff = (pr - ar).abs()

        # Flatten and select top-k positions
        flat_diff = diff.view(-1)
        top_indices = torch.topk(flat_diff, relu_budget[layer_name]).indices

        # Create new mask
        new_mask = torch.zeros_like(flat_diff, dtype=torch.uint8)
        new_mask[top_indices] = 1

        # Reshape to original activation shape
        updated_masks[layer_name] = new_mask.view_as(diff)

    return updated_masks

def apply_relu_with_mask(x, mask):
    return torch.where(mask.bool(), nn.ReLU()(x), x)




if __name__ == "__main__":
    activation_shape = (2, 3, 4)
    relu_budget = 5

    pr_activations = torch.randn(activation_shape)
    ar_activations = torch.randn(activation_shape)

    print("PR Activations:\n", pr_activations)
    print("AR Activations:\n", ar_activations)

    mask = initialize_relu_mask(activation_shape, relu_budget)
    print("Initial Mask:\n", mask)

    updated_mask = update_relu_mask(pr_activations, ar_activations, relu_budget)
    print("Updated Mask:\n", updated_mask)

    x = torch.randn(activation_shape)
    print("Input Tensor:\n", x)

    output = apply_relu_with_mask(x, updated_mask)
    print("Output after applying ReLU with mask:\n", output)