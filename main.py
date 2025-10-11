# calculate sensitivity 

# SENet: assign RELU counts using per layer budgets, along with RelU location

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sensitivity import  compute_relu_sensitivity, allocate_relu_budget
from evaluate import evaluate
from model import replace_relu_with_masked
from mask import initialize_relu_mask, update_relu_mask_nodict
from loss import distillation_loss

def main():
    
    ''' 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = resnet18(num_classes=10).to(device)
    model = load_model(resnet18(num_classes=10), "model/model_pr/pr_50_epoch_7.pth", device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # pr_model_loss, pr_model_accuracy = evaluate(model, testloader, criterion, device)
    # print(f"PR Model - Loss: {pr_model_loss:.4f}, Accuracy: {pr_model_accuracy:.2f}%")
    # print()

    avg_loss, accuracy = evaluate(model, testloader, criterion, device)

    sensitivity_scores = compute_relu_sensitivity(model, testloader, accuracy)
    print('Sensitivity score:\n', sensitivity_scores)
    print()


    # sensitivity_scores = {'relu': 0.03302499999999997, 
    #                       'layer1.0.relu': 0.42010000000000003, 
    #                       'layer1.1.relu': 0.5451, 
    #                       'layer2.0.relu': 0.27166250000000003, 
    #                       'layer2.1.relu': 0.37322500000000003, 
    #                       'layer3.0.relu': 0.04510000000000003, 
    #                       'layer3.1.relu': 0.03302499999999997, 
    #                       'layer4.0.relu': 0.02521249999999997, 
    #                       'layer4.1.relu': 0.009587499999999971}

    Total_ReLU_Budget = 50000
    relu_allocation = allocate_relu_budget(sensitivity_scores, Total_ReLU_Budget)
    print('ReLU allocation:\n', relu_allocation)
    print()

    # Training with distillation loss and ReLU masks
    model_pr = load_model(resnet18(num_classes=10), './model/model_epoch_21.pth', device)

    shape = get_activation_shapes(model_pr, torch.randn(1, 3, 32, 32))
    relu_masks = {
        layer_name: initialize_relu_mask(shape[layer_name], relu_allocation[layer_name])
        for layer_name in relu_allocation
    }

    replace_relu_with_masked(model_pr, relu_masks)

    for epoch in range(10):
        model_pr.train()
        model.eval()
        print(f"Epoch {epoch+1}")
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            
            ar_outputs, ar_activations = forward_with_activations(model, inputs)
            pr_outputs, pr_activations = forward_with_activations(model_pr, inputs)

            # Compute distillation loss
            loss = distillation_loss(
                    pr_outputs, targets,
                    pr_logits=pr_outputs,
                    ar_logits=ar_outputs,
                    pr_activations=pr_activations,
                    ar_activations=ar_activations
                )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update masks based on activation differences
            relu_masks = update_relu_mask_nodict(pr_activations, ar_activations, Total_ReLU_Budget)

        if epoch % 2 == 0:
            save_model(model_pr, f"model/model_pr/pr_50_epoch_{epoch+1}.pth")
    '''
    train_base_model()
    train_senet()
    test_senet_model()


def train_senet():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    avg_loss, accuracy = evaluate(model, testloader, criterion, device)

    sensitivity_scores = compute_relu_sensitivity(model, testloader, accuracy)
    print('Sensitivity score:\n', sensitivity_scores)
    print()
    Total_ReLU_Budget = 50000
    relu_allocation = allocate_relu_budget(sensitivity_scores, Total_ReLU_Budget)
    print('ReLU allocation:\n', relu_allocation)
    print()

    # Training with distillation loss and ReLU masks
    model_pr = load_model(resnet18(num_classes=10), './model/model_epoch_21.pth', device)

    shape = get_activation_shapes(model_pr, torch.randn(1, 3, 32, 32))
    relu_masks = {
        layer_name: initialize_relu_mask(shape[layer_name], relu_allocation[layer_name])
        for layer_name in relu_allocation
    }

    replace_relu_with_masked(model_pr, relu_masks)
    for epoch in range(10):
        model_pr.train()
        model.eval()
        print(f"Epoch {epoch+1}")
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            
            ar_outputs, ar_activations = forward_with_activations(model, inputs)
            pr_outputs, pr_activations = forward_with_activations(model_pr, inputs)

            # Compute distillation loss
            loss = distillation_loss(
                    pr_outputs, targets,
                    pr_logits=pr_outputs,
                    ar_logits=ar_outputs,
                    pr_activations=pr_activations,
                    ar_activations=ar_activations
                )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update masks based on activation differences
            relu_masks = update_relu_mask_nodict(pr_activations, ar_activations, Total_ReLU_Budget)

        if epoch % 2 == 0:
            save_model(model_pr, f"model/model_pr/pr_50_epoch_{epoch+1}.pth")


def test_senet_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(resnet18(num_classes=10), "model/model_pr/pr_50_epoch_7.pth", device)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    avg_loss, accuracy = evaluate(model, testloader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


def train_base_model():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(100):
        model.train()
        print(f"Epoch {epoch+1}/{100}")
        for idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        avg_loss, accuracy = evaluate(model, testloader, criterion, device)
    
        print(f"Epoch {epoch+1}: Accuracy = {accuracy:.2f}%")

        if epoch % 20 == 0:
            save_model(model, f"model/model_epoch_{epoch+1}.pth")



def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model


def forward_with_activations(model, x):
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_activation(name)))

    output = model(x)

    for hook in hooks:
        hook.remove()

    return output, activations


def get_activation_shapes(model, x):
    model.eval()
    activation_shapes = {}
    def get_shape(name):
        def hook(model, input, output):
            activation_shapes[name] = output.shape[1:]
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_shape(name)))

    with torch.no_grad():
        _ = model(x)

    for hook in hooks:
        hook.remove()

    return activation_shapes


if __name__ == "__main__":
    main()