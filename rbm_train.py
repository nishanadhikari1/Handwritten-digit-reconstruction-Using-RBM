"""
RBM Digit Generator — TRAIN
Approach: Train a standard RBM per digit class.
Each digit gets its OWN RBM trained only on that digit's images.
Generation = sample from that digit's RBM → guaranteed to look like that digit.

Run: python rbm_train.py
ETA: ~4-6 min on i7
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, gc

torch.set_num_threads(6)
torch.manual_seed(42)


# ──────────────────────────────────────────────────
#  One small RBM  (trained on ONE digit class only)
# ──────────────────────────────────────────────────
class RBM:
    def __init__(self, n_vis=784, n_hid=256):
        self.W  = torch.randn(n_hid, n_vis) * 0.01
        self.bv = torch.zeros(n_vis)
        self.bh = torch.zeros(n_hid)
        # momentum
        self.vW  = torch.zeros_like(self.W)
        self.vbv = torch.zeros_like(self.bv)
        self.vbh = torch.zeros_like(self.bh)

    def h_given_v(self, v):
        p = torch.sigmoid(v @ self.W.T + self.bh)
        return p, torch.bernoulli(p)

    def v_given_h(self, h):
        p = torch.sigmoid(h @ self.W + self.bv)
        return p, torch.bernoulli(p)

    def cd1(self, v0, lr=0.01, mom=0.9, wd=1e-4):
        # positive phase
        ph0, h0 = self.h_given_v(v0)
        # negative phase
        pv1, v1 = self.v_given_h(h0)
        ph1, _  = self.h_given_v(v1)

        bs = v0.size(0)
        gW  = (ph0.T @ v0 - ph1.T @ v1) / bs
        gbv = (v0 - v1).mean(0)
        gbh = (ph0 - ph1).mean(0)

        self.vW  = mom * self.vW  + lr * (gW  - wd * self.W)
        self.vbv = mom * self.vbv + lr * gbv
        self.vbh = mom * self.vbh + lr * gbh

        self.W  += self.vW
        self.bv += self.vbv
        self.bh += self.vbh

        return torch.mean((v0 - pv1) ** 2).item()

    @torch.no_grad()
    def sample(self, n=9, steps=2000):
        """Gibbs sample from this RBM"""
        v = torch.bernoulli(torch.rand(n, len(self.bv)) * 0.1)
        for _ in range(steps):
            _, h = self.h_given_v(v)
            pv, v = self.v_given_h(h)
        # return soft probabilities
        _, h = self.h_given_v(v)
        pv, _ = self.v_given_h(h)
        return pv

    def state_dict(self):
        return {'W': self.W, 'bv': self.bv, 'bh': self.bh}

    def load_state_dict(self, d):
        self.W  = d['W']
        self.bv = d['bv']
        self.bh = d['bh']


# ──────────────────────────────────────────────────
#  Train 10 separate RBMs (one per digit)
# ──────────────────────────────────────────────────
def train():
    print("\n╔════════════════════════════════════════════════╗")
    print("║  RBM Digit Generator — Training               ║")
    print("║  Strategy: 10 separate RBMs, one per digit    ║")
    print("║  Each RBM sees ONLY its digit → clean output  ║")
    print("╚════════════════════════════════════════════════╝\n")

    # Binary MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float().view(-1))
    ])
    full_ds = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)

    # Split dataset by digit label
    digit_data = {d: [] for d in range(10)}
    for img, lbl in full_ds:
        digit_data[lbl].append(img)

    for d in range(10):
        digit_data[d] = torch.stack(digit_data[d])   # (N, 784)
    print(f"  Samples per digit: ~{len(digit_data[0])}\n")

    rbms = {}
    EPOCHS = 40
    LR     = 0.005
    BATCH  = 128

    for digit in range(10):
        data = digit_data[digit]                      # (~6000, 784)
        n    = data.size(0)

        # init bv from data mean
        mean  = data.mean(0).clamp(0.01, 0.99)
        bv_init = torch.log(mean / (1 - mean))

        rbm = RBM(n_vis=784, n_hid=256)
        rbm.bv = bv_init.clone()

        losses = []
        for epoch in range(1, EPOCHS + 1):
            # shuffle
            idx  = torch.randperm(n)
            data = data[idx]
            ep_loss = 0.0
            for i in range(0, n, BATCH):
                batch    = data[i:i+BATCH]
                ep_loss += rbm.cd1(batch, lr=LR, mom=0.9, wd=1e-4)
            losses.append(ep_loss / (n // BATCH))

        final_loss = losses[-1]
        print(f"  Digit {digit} — final loss: {final_loss:.5f}")
        rbms[digit] = rbm.state_dict()
        del rbm, data
        gc.collect()

    # Save all 10 RBMs
    torch.save({'rbms': rbms, 'n_hid': 256, 'n_vis': 784}, 'rbm_model.pth')
    print("\n  ✓ rbm_model.pth saved  (10 RBMs inside)\n")

    # ── Verification: generate one per digit ──────────
    print("  Generating verification grid…", end='', flush=True)

    fig, axes = plt.subplots(3, 10, figsize=(16, 5), facecolor='#0d0d0d')

    # Real examples
    real = {}
    real_ds = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    for img, lbl in real_ds:
        if lbl not in real: real[lbl] = img
        if len(real) == 10: break

    for d in range(10):
        axes[0, d].imshow(real[d].view(28, 28).numpy(),
                          cmap='gray', vmin=0, vmax=1)
        axes[0, d].set_title(str(d), color='white',
                             fontsize=11, fontweight='bold')
        axes[0, d].axis('off')

    # Load & generate
    for d in range(10):
        rbm = RBM(n_vis=784, n_hid=256)
        rbm.load_state_dict(rbms[d])

        gen1 = rbm.sample(n=1, steps=2000)
        axes[1, d].imshow(gen1[0].view(28, 28).numpy(),
                          cmap='gray')
        axes[1, d].axis('off')

        gen2 = rbm.sample(n=1, steps=3000)
        axes[2, d].imshow(gen2[0].view(28, 28).numpy(),
                          cmap='gray')
        axes[2, d].axis('off')

    axes[0, 0].set_ylabel('Real',          color='#aaa', fontsize=9)
    axes[1, 0].set_ylabel('Gen (2k steps)', color='#aaa', fontsize=9)
    axes[2, 0].set_ylabel('Gen (3k steps)', color='#aaa', fontsize=9)

    plt.suptitle('Verification — each column is one digit (0–9)',
                 color='white', fontsize=12, y=1.01)
    fig.patch.set_facecolor('#0d0d0d')
    plt.tight_layout()
    plt.savefig('digit_check.png', dpi=130,
                bbox_inches='tight', facecolor='#0d0d0d')
    plt.close(); gc.collect()
    print(" done!")
    print("  ✓ digit_check.png  ← check this before running GUI\n")
    print("  Now run:  python rbm_generate.py\n")


if __name__ == '__main__':
    if os.path.exists('rbm_model.pth'):
        ck = torch.load('rbm_model.pth', map_location='cpu')
        if 'rbms' in ck and len(ck['rbms']) == 10:
            print("✓ Compatible rbm_model.pth found.")
            print("  Delete it to retrain. Running anyway…")
        else:
            print("⚠  Old model detected — deleting.")
            os.remove('rbm_model.pth')
            train()
    else:
        train()