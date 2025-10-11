import torch

def eval_subset(model, dataloader, criterion, device, subset_size=100):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i * dataloader.batch_size >= subset_size:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / (subset_size / dataloader.batch_size)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy