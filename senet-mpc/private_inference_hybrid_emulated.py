# private_inference_hybrid.py
import torch
import torch.nn.functional as F
import tenseal as ts
import numpy as np
import time
from model import Net
from torchvision import datasets, transforms

# -------------------------------
# 1. Polynomial ReLU (FHE-friendly)
# -------------------------------
def relu_poly(x):
    return (x * x) * 0.001 + x * 0.3 + 0.05

# -------------------------------
# 2. Device and data setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# -------------------------------
# 3. Load trained model
# -------------------------------
model = Net().to(device)
model.load_state_dict(torch.load("cifar_model.pth", map_location=device))
model.eval()

# -------------------------------
# 4. TenSEAL CKKS context
# -------------------------------
# Adjusted to prevent scale overflow:
# - initial global_scale smaller
# - coeff_mod_bit_sizes supports multiplicative depth (FC1*poly*FC2)
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]  # extra level for depth
)
context.global_scale = 2**25  # smaller initial scale to accommodate multiplications
context.generate_galois_keys()

# -------------------------------
# 5. Extract FC weights and biases
# -------------------------------
w1 = (model.fc1.weight.data.cpu().numpy().astype(np.float64) / 10).tolist()
b1 = (model.fc1.bias.data.cpu().numpy().astype(np.float64) / 10).tolist()
w2 = (model.fc2.weight.data.cpu().numpy().astype(np.float64) / 10).tolist()
b2 = (model.fc2.bias.data.cpu().numpy().astype(np.float64) / 10).tolist()

# -------------------------------
# 6. FHE Private Inference
# -------------------------------
total = correct = 0
start_time_total = time.time()
print("TenSEAL: starting hybrid encrypted inference...")

for idx, (images, labels) in enumerate(testloader):
    images, labels = images.to(device), labels.to(device)

    # ---------------------------
    # 6a. Plaintext conv + pool
    # ---------------------------
    t0 = time.time()
    with torch.no_grad():
        x = F.relu(model.conv1(images))
        x = model.pool(x)
        x = F.relu(model.conv2(x))
        x = model.pool(x)
        x_flat = x.view(-1).cpu().numpy().astype(np.float64)
    t1 = time.time()

    # ---------------------------
    # 6b. Encrypt flattened input
    # ---------------------------
    enc_x = ts.ckks_vector(context, x_flat.tolist())
    t2 = time.time()

    # ---------------------------
    # 6c. Encrypted FC1 + polynomial ReLU
    # ---------------------------
    t_fc1_start = time.time()
    enc_fc1 = enc_x.mm(np.array(w1).T)
    enc_fc1 += b1
    enc_fc1 = relu_poly(enc_fc1)
    t_fc1_end = time.time()

    # ---------------------------
    # 6d. Encrypted FC2
    # ---------------------------
    t_fc2_start = time.time()
    enc_out = enc_fc1.mm(np.array(w2).T)
    enc_out += b2
    t_fc2_end = time.time()

    # ---------------------------
    # 6e. Decrypt output
    # ---------------------------
    dec_out = enc_out.decrypt()
    pred_class = int(torch.argmax(torch.tensor(dec_out)))

    total += 1
    correct += (pred_class == labels.item())

    # ---------------------------
    # Logging per image
    # ---------------------------
    print(f"[{idx+1}/{len(testloader)}] "
          f"Plain conv: {t1-t0:.4f}s, "
          f"Encrypt: {t2-t1:.4f}s, "
          f"FC1+ReLU: {t_fc1_end-t_fc1_start:.4f}s, "
          f"FC2: {t_fc2_end-t_fc2_start:.4f}s, "
          f"Pred: {pred_class}, Label: {labels.item()}")

end_time_total = time.time()

# -------------------------------
# 7. Print final results
# -------------------------------
print(f"\nFHE Private Inference Accuracy: {100*correct/total:.2f}%")
print(f"Total inference time for {total} images: {end_time_total-start_time_total:.2f} sec")
print(f"Average time per image: {(end_time_total-start_time_total)/total:.4f} sec")
