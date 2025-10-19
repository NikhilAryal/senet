import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net

# -------------------------------
# Placeholder for future ReLU masking
# -------------------------------
def mask(x):
    """
    Currently does nothing.
    Later, this function can disable ReLUs that are rarely active
    or apply a mask to speed up FHE inference.
    """
    return x

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Initialize model, loss, optimizer
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (5 epochs)
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')

    # Evaluate on test set
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # -------------------------------
    # Save the trained model
    # -------------------------------
    torch.save(model.state_dict(), "cifar_model.pth")
    print("Model saved as cifar_model.pth")

if __name__ == "__main__":
    main()
