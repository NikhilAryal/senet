import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from mask import inject_mask, LocalMask

class MaskedReLU(nn.Module):
    def __init__(self, mask):
        super(MaskedReLU, self).__init__()
        self.mask = mask  

    def forward(self, x):
        # Apply ReLU only where mask == 1
        return torch.where(self.mask.bool(), nn.functional.relu(x), x)


# Test class to update Resnet , Unused
class CustomResNet(nn.Module):
    def __init__(self, base_model, relu_masks):
        super(CustomResNet, self).__init__(block = BasicBlock, layers=[2, 2, 2, 2], num_classes=10)
        self.model = base_model
        self.relu_masks = relu_masks  # Dictionary of layer_name -> mask

        # Replace ReLU layers with masked versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(self.model, name, MaskedReLU())

    def forward(self, x):
        '''
        # You need to pass masks to each MaskedReLU layer manually
        # This requires modifying the forward method of the base model
        # Example:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x, self.relu_masks['relu'])  # Masked ReLU
        # Continue forward pass...
        return x

        '''

        x = self.forward(x)

        x = self.conv1(x)         # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # [B, 64, H/4, W/4]

        x = self.layer1(x)        # [B, 64, H/4, W/4]
        x = self.layer2(x)        # [B, 128, H/8, W/8]
        x = self.layer3(x)        # [B, 256, H/16, W/16]
        x = self.layer4(x)        # [B, 512, H/32, W/32]

        x = self.avgpool(x)       # [B, 512, 1, 1]
        x = torch.flatten(x, 1)   # [B, 512]
        x = self.fc(x)

    

def MaskedReluActivation(x, input, output, mask):
    print(f"Inside MaskedReLU forward")
    return torch.where(mask.bool(), nn.ReLU()(x), x)



def MaskedReluHook(model, input, output, mask):
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            print(f"Module name: {name}")
            layer_mask = mask[name]
            hook_with_mask = inject_mask(layer_mask)(MaskedReluActivation)
            module.register_forward_hook(hook_with_mask)
    return model


# replace relu with mask after training AR model
def replace_relu_with_masked(model, relu_masks):
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            masked_relu = MaskedReLU(relu_masks)
            # Replace the module with masked version
            set_module_by_name(model, name, masked_relu)
            # Store mask reference if needed
            masked_relu.mask = relu_masks.get(name, None)


def set_module_by_name(model, name, new_module):
    """
    Replace a module in a PyTorch model by its dotted name.

    Args:
        model (nn.Module): The model containing the module.
        name (str): Dotted name of the module (e.g., 'layer1.0.relu').
        new_module (nn.Module): The new module to insert.
    """
    parts = name.split('.')
    submodule = model
    for part in parts[:-1]:
        submodule = getattr(submodule, part)
    setattr(submodule, parts[-1], new_module)