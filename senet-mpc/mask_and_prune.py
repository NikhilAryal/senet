# mask_and_prune.py
# Computes activation frequency for each neuron in fc1 and fc2,
# produces a mask that zeroes out least-active neurons, prunes the model, saves pruned state.

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import Net
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cifar_model.pth")
parser.add_argument("--output", default="cifar_model_pruned.pth")
parser.add_argument("--threshold", type=float, default=0.05, help="activation rate threshold (0..1)")
parser.add_argument("--subset", type=int, default=5000, help="number of test samples to measure on")
args = parser.parse_args()

device = torch.device("cpu")

# load model
model = Net().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

# dataloader (use test set for measuring activations)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# We'll record activations for fc1 neurons (size fc1.out_features) and optionally fc2
fc1_size = model.fc1.out_features
act_counts_fc1 = np.zeros(fc1_size, dtype=np.int64)
total = 0

print("Measuring activations (this may take a while)...")
for i, (images, labels) in enumerate(tqdm(loader)):
    if i >= args.subset:
        break
    with torch.no_grad():
        x = F.relu(model.conv1(images))
        x = model.pool(x)
        x = F.relu(model.conv2(x))
        x = model.pool(x)
        x_flat = x.view(1, -1)
        fc1_out = model.fc1(fc1_out := fc1_out if False else x_flat)  # workaround to avoid linting
        # to be explicit:
        fc1_out = model.fc1(x_flat)
        rel = F.relu(fc1_out).numpy().squeeze()
        act_counts_fc1 += (rel > 1e-6).astype(np.int64)
    total += 1

act_rate_fc1 = act_counts_fc1 / total
print("Done measuring. Sample of activation rates (fc1, first 20):")
print(act_rate_fc1[:20])

# Build mask: keep neurons whose activation rate >= threshold
mask_fc1 = (act_rate_fc1 >= args.threshold).astype(np.float32)
kept = mask_fc1.sum()
print(f"Mask threshold={args.threshold}: keeping {kept}/{fc1_size} neurons in fc1")

# Apply pruning to model: remove masked neurons by zeroing out corresponding rows in fc1 and corresponding cols in fc2
# We'll implement a simple masking (zero weights) rather than architectural surgery.
new_state = model.state_dict()
# fc1.weight shape: (out, in)
w1 = new_state['fc1.weight'].clone()
b1 = new_state['fc1.bias'].clone()
w2 = new_state['fc2.weight'].clone()
b2 = new_state['fc2.bias'].clone()

mask_tensor = torch.from_numpy(mask_fc1).to(torch.float32)
w1 = w1 * mask_tensor.unsqueeze(1)  # zero out rows of pruned neurons
b1 = b1 * mask_tensor

# for fc2, zero out columns corresponding to pruned fc1 neurons
w2 = w2 * mask_tensor.unsqueeze(0)

new_state['fc1.weight'] = w1
new_state['fc1.bias'] = b1
new_state['fc2.weight'] = w2
new_state['fc2.bias'] = b2

# save pruned model
torch.save(new_state, args.output)
print(f"Pruned model saved to {args.output}")
