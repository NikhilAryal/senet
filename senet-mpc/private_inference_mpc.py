import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import crypten
# Initialize CrypTen
crypten.init()
import crypten.communicator as comm
from model import Net
import time
import numpy as np


# -------------------------------
# 1. Setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


rank = comm.get().get_rank()
world_size = comm.get().get_world_size()
print(f"I am party {rank} out of {world_size}")

# -------------------------------
# 2. Load model
# -------------------------------
model = Net().to(DEVICE)
model.load_state_dict(torch.load("cifar_model.pth", map_location=DEVICE))
model.eval()

# -------------------------------
# 3. Data
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
NUM_SAMPLES = len(testset)

# -------------------------------
# 4. Collect conv outputs
# -------------------------------
conv_vectors = []
labels = []
print("Collecting conv-layer outputs (plaintext) for MPC FC inference...")
for idx, (img, label) in enumerate(loader):
    if idx >= NUM_SAMPLES:
        break
    with torch.no_grad():
        x = F.relu(model.conv1(img.to(DEVICE)))
        x = model.pool(x)
        x = F.relu(model.conv2(x))
        x = model.pool(x)
        x_flat = x.view(-1).cpu()  # flatten
        conv_vectors.append(x_flat)
        labels.append(int(label))
print(f"Collected {len(conv_vectors)} vectors.")

# -------------------------------
# 5. Extract FC weights and biases
# -------------------------------
# Convert FC weights to numpy arrays for encryption
w1 = model.fc1.weight.data.cpu().numpy()
b1 = model.fc1.bias.data.cpu().numpy()
w2 = model.fc2.weight.data.cpu().numpy()
b2 = model.fc2.bias.data.cpu().numpy()

# -------------------------------
# 6. Private inference function
# -------------------------------
def private_fc_relu_inference(conv_vectors):
    preds = []
    times = []

    # Convert FC weights/biases to crypten tensors once
    w1_enc = crypten.cryptensor(torch.tensor(w1.T, dtype=torch.float32))
    b1_enc = crypten.cryptensor(torch.tensor(b1, dtype=torch.float32))
    w2_enc = crypten.cryptensor(torch.tensor(w2.T, dtype=torch.float32))
    b2_enc = crypten.cryptensor(torch.tensor(b2, dtype=torch.float32))

    for vec in conv_vectors:
        t0 = time.time()
        # Secret-share input vector
        x_enc = crypten.cryptensor(vec, src=0)

        # FC1 + ReLU
        fc1_enc = x_enc @ w1_enc
        fc1_enc += b1_enc
        fc1_enc = fc1_enc.relu()

        # FC2
        fc2_enc = fc1_enc @ w2_enc
        fc2_enc += b2_enc

        # Decrypt output
        out = fc2_enc.get_plain_text()
        pred_class = int(torch.argmax(out))
        preds.append(pred_class)
        times.append(time.time() - t0)

    return preds, times

# -------------------------------
# 7. Run private inference
# -------------------------------
preds, times = private_fc_relu_inference(conv_vectors)

# -------------------------------
# 8. Evaluate accuracy
# -------------------------------
correct = sum([p == l for p, l in zip(preds, labels)])
print(f"\nCrypTen MPC private inference accuracy on {NUM_SAMPLES} samples: {100*correct/NUM_SAMPLES:.2f}%")
print(f"Average inference time per sample: {np.mean(times):.4f}s")