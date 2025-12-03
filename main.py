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
    # train_base_model()
    # train_senet()
    # test_senet_model()
    # train_base_vgg16()
    # train_base_inception()
    train_senet_inception()


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


def train_base_inception():
    # pass
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #   Resize to 32x32 for VGG
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Modify First Layer (Input): Change 3 channels -> 1 channel
    # In torchvision's GoogLeNet, conv1 is a 'BasicConv2d' block, and the actual conv is 'conv'
    # model = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT)
    # model.aux_logits = False  # Disable auxiliary classifiers
    # model.transform_input = False
    # model.conv1.conv = nn.Conv2d(in_channels=1, out_channels=64, 
    #                             kernel_size=7, stride=2, padding=3, bias=False)

    # # Modify Last Layer (Output): Change 1000 classes -> 10 classes
    # model.fc = nn.Linear(in_features=1024, out_features=10)

    # model = model.to(device)

    # 4. Training Pipeline
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

    # Modify First Layer (Input): Change 3 channels -> 1 channel
    # Original: Conv2d(3, 64, kernel_size=(3, 3), ...)
    model.features[0] = nn.Conv2d(in_channels=1, out_channels=64, 
                                kernel_size=3, padding=1)

    # Modify Last Layer (Output): Change 1000 classes -> 10 classes
    # Access the classifier sequential block (index 6 is the final linear layer)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10)

    model = model.to(device)

    # 4. Training Pipeline
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for epoch in range(15):
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.4f} | Acc: {100 * correct / total:.2f}%")
        
        avg_loss, accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}: Accuracy = {accuracy:.2f}%")

        if epoch % 10 == 0:
            save_model(model, f"model/model_inception_epoch_{epoch+1}.pth")


def train_senet_inception():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #   Resize to 32x32 for VGG
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Modify First Layer (Input): Change 3 channels -> 1 channel
    # In torchvision's GoogLeNet, conv1 is a 'BasicConv2d' block, and the actual conv is 'conv'
    # model = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT)
    # model.aux_logits = False  # Disable auxiliary classifiers
    # model.transform_input = False
    # model.conv1.conv = nn.Conv2d(in_channels=1, out_channels=64, 
    #                             kernel_size=7, stride=2, padding=3, bias=False)

    # # Modify Last Layer (Output): Change 1000 classes -> 10 classes
    # model.fc = nn.Linear(in_features=1024, out_features=10)

    # model = model.to(device)

    # # 4. Training Pipeline
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

    # Modify First Layer (Input): Change 3 channels -> 1 channel
    # Original: Conv2d(3, 64, kernel_size=(3, 3), ...)
    model.features[0] = nn.Conv2d(in_channels=1, out_channels=64, 
                                kernel_size=3, padding=1)

    # Modify Last Layer (Output): Change 1000 classes -> 10 classes
    # Access the classifier sequential block (index 6 is the final linear layer)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10)

    model = model.to(device)

    # 4. Training Pipeline
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    avg_loss, accuracy = evaluate(model, test_loader, criterion, device)

    sensitivity_scores = compute_relu_sensitivity(model, test_loader, accuracy)
    print('Sensitivity score:\n', sensitivity_scores)
    print()
    Total_ReLU_Budget = 50000
    relu_allocation = allocate_relu_budget(sensitivity_scores, Total_ReLU_Budget)
    print('ReLU allocation:\n', relu_allocation)
    print()

    # Training with distillation loss and ReLU masks
    model_pr = load_model(model, './model/model_vgg16_epoch_1.pth', device)

    shape = get_activation_shapes(model_pr, torch.randn(1, 1, 32, 32).to(device))
    relu_masks = {
        layer_name: initialize_relu_mask(shape[layer_name], relu_allocation[layer_name])
        for layer_name in relu_allocation
    }

    replace_relu_with_masked(model_pr, relu_masks)
    for epoch in range(10):
        model_pr.train()
        model.eval()
        print(f"Epoch {epoch+1}")
        for inputs, targets in train_loader:
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

        if (epoch+1) % 5 == 0:
            save_model(model_pr, f"model/model_pr/model_inception_pr_10_epoch_{epoch+1}.pth")

# utils
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