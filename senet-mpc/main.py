import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pretend there is no GPU
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18

import crypten
import crypten.communicator as comm
import crypten.mpc as mpc

DEVICE = torch.device("cpu")

def load_resnet_model(weights_path: str) -> torch.nn.Module:
    model = resnet18(weights=None, num_classes=10).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys: {missing}\nunexpected keys: {unexpected}")
    model.eval()
    return model


def build_data_loader(batch_size: int = 1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = datasets.CIFAR10(root="../data", train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return testset, loader


def collect_conv_vectors(model: torch.nn.Module, loader, limit: int = None):
    conv_vectors = []
    labels = []
    for idx, (img, label) in enumerate(loader):
        if limit is not None and idx >= limit:
            break
        with torch.no_grad():
            x = img.to(DEVICE)
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x_flat = torch.flatten(x, 1).squeeze(0).cpu()
            conv_vectors.append(x_flat)
            labels.append(int(label))
    return conv_vectors, labels


def private_fc_inference(conv_vectors, w_fc, b_fc):
    preds = []
    times = []

    w_enc = crypten.cryptensor(torch.tensor(w_fc.T, dtype=torch.float32))
    b_enc = crypten.cryptensor(torch.tensor(b_fc, dtype=torch.float32))

    for vec in conv_vectors:
        t0 = time.time()
        x_enc = crypten.cryptensor(vec, src=0)

        logits_enc = x_enc @ w_enc
        logits_enc += b_enc

        out = logits_enc.get_plain_text()
        pred_class = int(torch.argmax(out))
        preds.append(pred_class)
        times.append(time.time() - t0)

    return preds, times


def main(num_samples: int = 1000):
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    print(f"I am party {rank} out of {world_size}")

    project_root = os.path.dirname(os.path.dirname(__file__))
    weights_path = os.path.join(project_root, "model", "model_epoch_21.pth")
    print("Loading weights from:", weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    model = load_resnet_model(weights_path)

    testset, loader = build_data_loader(batch_size=1)
    num_samples = min(num_samples, len(testset))
    print(f"Collecting features for {num_samples} samples...")
    conv_vectors, labels = collect_conv_vectors(model, loader, limit=num_samples)

    w_fc = model.fc.weight.data.cpu().numpy()
    b_fc = model.fc.bias.data.cpu().numpy()

    print("Running CrypTen MPC for final FC layer...")
    preds, times = private_fc_inference(conv_vectors, w_fc, b_fc)

    correct = sum(int(p == l) for p, l in zip(preds, labels))
    acc = 100.0 * correct / num_samples if num_samples else 0.0
    print(f"\nMPC private inference accuracy on {num_samples} samples: {acc:.2f}%")
    print(f"Average inference time per sample: {np.mean(times):.4f}s")


@mpc.run_multiprocess(world_size=3)
def run():
    ns = int(os.environ.get("NUM_SAMPLES", "50"))
    main(num_samples=ns)


if __name__ == "__main__":
    run()
