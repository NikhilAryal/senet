# private_inference_tenseal.py
# Measure TenSEAL encrypted linear layer timings (FC1 + FC2).
# Conv layers remain plaintext, same as in earlier experiments.

import time
import torch
import torch.nn.functional as F
import tenseal as ts
import numpy as np
from torchvision import datasets, transforms
from model import Net

DEVICE = torch.device("cpu")
MODEL_PATH = "cifar_model.pth"
NUM_SAMPLES = 100  # number of test images to profile (reduce for quicker runs)

# load model
model = Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# prepare test loader (use small subset)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# TenSEAL CKKS context (safe, medium params)
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60,40,40,40,60])
context.global_scale = 2**30
context.generate_galois_keys()

# weights for FC layers (as numpy)
w1 = model.fc1.weight.data.cpu().numpy().astype(np.float64)  # (128, 2048)
b1 = model.fc1.bias.data.cpu().numpy().astype(np.float64)
w2 = model.fc2.weight.data.cpu().numpy().astype(np.float64)  # (10, 128)
b2 = model.fc2.bias.data.cpu().numpy().astype(np.float64)

times = []
acc = 0
count = 0
print("TenSEAL: profiling encrypted linear layers (FC1 and FC2)...")
for i, (img, label) in enumerate(loader):
    if i >= NUM_SAMPLES:
        break
    with torch.no_grad():
        x = F.relu(model.conv1(img))
        x = model.pool(x)
        x = F.relu(model.conv2(x))
        x = model.pool(x)
        x_flat = x.view(-1).cpu().numpy().astype(np.float64)

    t0 = time.time()
    enc_x = ts.ckks_vector(context, x_flat.tolist())
    t_enc = time.time()

    t0_fc1 = time.time()
    enc_fc1 = enc_x.mm(w1.T.tolist())
    enc_fc1 += b1.tolist()
    t1_fc1 = time.time()

    t0_fc2 = time.time()
    enc_fc2 = enc_fc1.mm(w2.T.tolist())
    enc_fc2 += b2.tolist()
    t1_fc2 = time.time()

    t_dec = time.time()
    dec = enc_fc2.decrypt()
    t_dec2 = time.time()

    time_rec = {
        "encryption": t_enc - t0,
        "fc1_enc_time": t1_fc1 - t0_fc1,
        "fc2_enc_time": t1_fc2 - t0_fc2,
        "decrypt_time": t_dec2 - t_dec,
        "total": t_dec2 - t0
    }
    times.append(time_rec)
    pred = int(torch.argmax(torch.tensor(dec)))
    acc += (pred == int(label))
    count += 1
    print(f"[{i+1}/{NUM_SAMPLES}] enc_fc1: {time_rec['fc1_enc_time']:.3f}s, enc_fc2: {time_rec['fc2_enc_time']:.3f}s, total: {time_rec['total']:.3f}s, pred={pred}, label={int(label)}")

# summary
import numpy as np
arr = np.array([[t["encryption"], t["fc1_enc_time"], t["fc2_enc_time"], t["decrypt_time"], t["total"]] for t in times])
print("\nTenSEAL summary (seconds):")
print("mean:", arr.mean(axis=0))
print("median:", np.median(arr, axis=0))
print(f"accuracy on {count} samples (TenSEAL decrypted logits): {100*acc/count:.2f}%")
