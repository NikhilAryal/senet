
import torch.nn.functional as F
import torch.nn as nn

def distillation_loss(pr_output, targets, pr_logits, ar_logits, pr_activations, ar_activations, temperature=4.0, alpha=0.5, beta=1.0):
    """
    Computes the combined distillation loss for training the PR model.

    Args:
        pr_output (Tensor): Output logits from PR model.
        targets (Tensor): Ground truth labels.
        pr_logits (Tensor): Logits from PR model.
        ar_logits (Tensor): Logits from AR model.
        pr_activations (dict): Intermediate activations from PR model.
        ar_activations (dict): Intermediate activations from AR model.
        temperature (float): Temperature for softening logits.
        alpha (float): Weight for cross-entropy loss.
        beta (float): Weight for activation similarity loss.

    Returns:
        total_loss (Tensor): Combined distillation loss.
    """
    # Cross-entropy loss with ground truth
    ce_loss = nn.CrossEntropyLoss()(pr_output, targets)

    # KL divergence between softened logits
    kl_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(pr_logits / temperature, dim=1),
        F.softmax(ar_logits / temperature, dim=1)
    ) * (temperature ** 2)

    # Activation similarity loss (PRAM loss)
    pram_loss = 0.0
    for layer in pr_activations:
        pram_loss += F.mse_loss(pr_activations[layer], ar_activations[layer])

    # Combine all losses
    total_loss = alpha * ce_loss + (1 - alpha) * kl_loss + beta * pram_loss
    return total_loss
