# private_inference_hybrid_mpc_fhe.py
# Hybrid Private Inference:
# - Conv layers: plaintext (or TenSEAL CKKS encrypted)
# - FC layers: MPC via CrypTen

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import tenseal as ts
import crypten
import crypten.communicator as comm
import numpy as np
import time
from model import Net

# -------------------------------
# 1. Device and data setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# -------------------------------
# 2. Load trained model
# -------------------------------
model = Net().to(DEVICE)
model.load_state_dict(torch.load("cifar_model.pth", map_location=DEVICE))
model.eval()

# -------------------------------
# 3. Extract FC weights and biases
# -------------------------------
w1 = model.fc1.weight.data.cpu().numpy().astype(np.float32)
b1 = model.fc1.bias.data.cpu().numpy().astype(np.float32)
w2 = model.fc2.weight.data.cpu().numpy().astype(np.float32)
b2 = model.fc2.bias.data.cpu().numpy().astype(np.float32)

# -------------------------------
# 4. TenSEAL CKKS context (optional conv encryption)
# -------------------------------
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**20
context.generate_galois_keys()

# -------------------------------
# 5. CrypTen initialization
# -------------------------------
crypten.init()
# comm.get().set_default_group()

# -------------------------------
# 6. Helper: secure FC inference
# -------------------------------
def secure_fc_inference(conv_vectors):
    preds = []
    times = []

    w1_enc = crypten.cryptensor(w1.T)  # FC1 weights
    b1_enc = crypten.cryptensor(b1)
    w2_enc = crypten.cryptensor(w2.T)  # FC2 weights
    b2_enc = crypten.cryptensor(b2)

    for x_flat in conv_vectors:
        t0 = time.time()
        # convert input to CrypTen tensor
        x_enc = crypten.cryptensor(x_flat)

        # FC1 + ReLU
        fc1_enc = x_enc @ w1_enc
        fc1_enc += b1_enc
        fc1_enc = fc1_enc.relu()

        # FC2
        fc2_enc = fc1_enc @ (w2_enc)
        fc2_enc += b2_enc

        # decrypt prediction
        pred = int(torch.argmax(fc2_enc.get_plain_text()))
        t1 = time.time()

        preds.append(pred)
        times.append(t1 - t0)

    return preds, times

# -------------------------------
# 7. Collect conv outputs (plaintext)
# -------------------------------
NUM_SAMPLES = 50
conv_vectors = []
labels = []

print("Collecting conv-layer outputs (plaintext) for MPC FC inference...")
for i, (images, label) in enumerate(testloader):
    if i >= NUM_SAMPLES:
        break
    with torch.no_grad():
        images = images.to(DEVICE)
        x = F.relu(model.conv1(images))
        x = model.pool(x)
        x = F.relu(model.conv2(x))
        x = model.pool(x)
        x_flat = x.view(-1).cpu().numpy().astype(np.float32)
        conv_vectors.append(x_flat)
        labels.append(int(label))
print(f"Collected {len(conv_vectors)} vectors.")

# -------------------------------
# 8. Run secure FC inference (CrypTen)
# -------------------------------
preds, times = secure_fc_inference(conv_vectors)

# -------------------------------
# 9. Accuracy and timing
# -------------------------------
correct = sum([p == l for p, l in zip(preds, labels)])
print(f"\nCrypTen MPC private inference accuracy on {NUM_SAMPLES} samples: {100*correct/NUM_SAMPLES:.2f}%")
print(f"Average inference time per sample: {np.mean(times):.4f}s")
