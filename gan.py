"""
gan.py - Conditional GAN (cGAN) for handwritten digits.

Everything the project needs lives in this one file:

    * Generator / Discriminator models
    * train_gan()        -> train on MNIST (Google Colab friendly) and save the model
    * load_generator()   -> load the saved model for the Streamlit app
    * generate()         -> turn digit labels + noise into 28x28 images

Quick start (Google Colab):

    from gan import train_gan
    train_gan(epochs=30)          # saves models/generator.pt
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

LATENT_DIM = 100   # size of the random noise vector
NUM_CLASSES = 10   # digits 0-9
IMG_SIZE = 28      # MNIST images are 28 x 28


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class Generator(nn.Module):
    """noise (100) + digit label (one-hot, 10)  ->  28x28 image in [-1, 1]."""

    def __init__(self, latent_dim: int = LATENT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 7  -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),                # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        x = self.fc(torch.cat([z, onehot], dim=1)).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """image + digit label  ->  one score (real vs. fake)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1 + num_classes, 64, 4, 2, 1),            # 28 -> 14
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),            # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, img: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        label_maps = onehot[:, :, None, None].expand(-1, -1, img.size(2), img.size(3))
        return self.net(torch.cat([img, label_maps], dim=1))


# --------------------------------------------------------------------------- #
# Small helpers (used by training and by the Streamlit app)
# --------------------------------------------------------------------------- #
def digits_to_onehot(digits: Sequence[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """[3, 7] -> one-hot float tensor of shape (2, 10)."""
    return F.one_hot(torch.as_tensor(list(digits), dtype=torch.long), num_classes).float()


def make_noise(n: int, latent_dim: int = LATENT_DIM, seed: int | None = None) -> torch.Tensor:
    """Random noise vectors. The same seed always gives the same noise."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(n, latent_dim, generator=gen)


@torch.no_grad()
def generate(generator: Generator, onehot: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    """Generate images. Returns a uint8 array of shape (n, 28, 28), 0 = black, 255 = white."""
    was_training = generator.training
    generator.eval()
    device = next(generator.parameters()).device
    out = generator(z.to(device), onehot.to(device)).cpu().numpy()[:, 0]
    generator.train(was_training)
    return ((out + 1) * 127.5).clip(0, 255).astype(np.uint8)


def tile_images(images: np.ndarray, ncols: int, pad: int = 2) -> np.ndarray:
    """Arrange (n, H, W) images into one (rows*H, cols*W) picture."""
    n, h, w = images.shape
    nrows = int(np.ceil(n / ncols))
    canvas = np.zeros((nrows * (h + pad) + pad, ncols * (w + pad) + pad), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = img
    return canvas


def _progress_image(snapshots: dict[int, np.ndarray], scale: int = 3) -> Image.Image:
    """One row per saved epoch, one column per digit 0-9: shows the GAN learning."""
    size = IMG_SIZE * scale
    margin, gap = 80, 4
    width = margin + NUM_CLASSES * (size + gap)
    height = len(snapshots) * (size + gap) + gap
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for row, (epoch, imgs) in enumerate(snapshots.items()):
        y = gap + row * (size + gap)
        draw.text((8, y + size // 2 - 5), f"epoch {epoch}", fill=170)
        for col, img in enumerate(imgs):
            tile = Image.fromarray(img).resize((size, size), Image.BICUBIC)
            canvas.paste(tile, (margin + col * (size + gap), y))
    return canvas


def _load_mnist(data_dir: str):
    from torchvision import datasets, transforms  # imported here: only needed for training

    return datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())


def _save_generator(generator: Generator, path: Path, epochs_done: int) -> None:
    state = {k: v.detach().cpu() for k, v in generator.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "latent_dim": generator.latent_dim,
            "num_classes": generator.num_classes,
            "epochs": epochs_done,
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_gan(
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-4,
    latent_dim: int = LATENT_DIM,
    model_dir: str = "models",
    assets_dir: str = "assets",
    data_dir: str = "data",
    dataset=None,
    device: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train the conditional GAN and save everything the app needs.

    Saves
    -----
    models/generator.pt            the trained generator (the app loads this)
    assets/samples.png             a grid of generated digits
    assets/training_progress.png   how the digits improve epoch by epoch

    Parameters
    ----------
    epochs : 30 epochs take only a few minutes on a free Colab GPU.
    dataset : optional. Any torch Dataset returning (image[1,28,28] in 0..1, label).
              Leave as None to download MNIST automatically.

    Returns
    -------
    dict with the loss history: {"d_loss": [...], "g_loss": [...]}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, assets_dir = Path(model_dir), Path(assets_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataset if dataset is not None else _load_mnist(data_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # keeps BatchNorm happy
        num_workers=2 if device == "cuda" else 0,
        pin_memory=device == "cuda",
    )

    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    # Fixed noise: one image per digit, used to watch the GAN improve over time.
    fixed_z = make_noise(NUM_CLASSES, latent_dim, seed=123)
    fixed_y = digits_to_onehot(range(NUM_CLASSES))
    snap_epochs = sorted({e for e in (1, 3, 5, 10, 20, 30, 50, 100) if e <= epochs} | {epochs})
    snapshots: dict[int, np.ndarray] = {}
    history = {"d_loss": [], "g_loss": []}

    for epoch in range(1, epochs + 1):
        start = time.time()
        d_sum = g_sum = 0.0
        for real, labels in loader:
            real = real.to(device) * 2 - 1                 # [0,1] -> [-1,1]
            y = F.one_hot(labels.to(device), NUM_CLASSES).float()
            b = real.size(0)

            # Fake images: random noise + random digit labels
            z = torch.randn(b, latent_dim, device=device)
            fake_y = F.one_hot(torch.randint(0, NUM_CLASSES, (b,), device=device), NUM_CLASSES).float()
            fake = G(z, fake_y)

            # 1) Train the Discriminator: real -> 1 (smoothed to 0.9), fake -> 0
            loss_d = bce(D(real, y), torch.full((b, 1), 0.9, device=device)) + bce(
                D(fake.detach(), fake_y), torch.zeros(b, 1, device=device)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # 2) Train the Generator: try to make the Discriminator say "real"
            loss_g = bce(D(fake, fake_y), torch.ones(b, 1, device=device))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            d_sum += loss_d.item()
            g_sum += loss_g.item()

        history["d_loss"].append(d_sum / len(loader))
        history["g_loss"].append(g_sum / len(loader))
        _save_generator(G, model_dir / "generator.pt", epoch)  # safe even if Colab disconnects

        if epoch in snap_epochs:
            snapshots[epoch] = generate(G, fixed_y, fixed_z)
        if verbose:
            print(
                f"Epoch {epoch:>3}/{epochs} | D loss {history['d_loss'][-1]:.3f} | "
                f"G loss {history['g_loss'][-1]:.3f} | {time.time() - start:.1f}s"
            )

    # Save preview pictures (great for the README and LinkedIn)
    z = make_noise(60, latent_dim, seed=7)
    y = digits_to_onehot([i % NUM_CLASSES for i in range(60)])
    grid = tile_images(generate(G, y, z), ncols=10)
    Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.BICUBIC).save(
        assets_dir / "samples.png"
    )
    _progress_image(snapshots).save(assets_dir / "training_progress.png")

    if verbose:
        print(f"\nDone! Model saved to {model_dir / 'generator.pt'}")
    return history


# --------------------------------------------------------------------------- #
# Loading (used by the Streamlit app)
# --------------------------------------------------------------------------- #
def load_generator(path: str | Path = "models/generator.pt", device: str = "cpu") -> Generator:
    """Load a trained generator saved by train_gan(). Works on CPU-only machines."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Generator(ckpt["latent_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.trained_epochs = ckpt.get("epochs", "?")  # shown in the app's model card
    return model.to(device).eval()
