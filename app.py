"""
app.py - Streamlit UI for the handwritten-digit GAN.

Run:  streamlit run app.py
"""
import io
import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

import gan

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "generator.pt"
ASSETS = ROOT / "assets"

INK_COLORS = {
    "Neon Cyan": (34, 211, 238),
    "Hot Pink": (244, 114, 182),
    "Lime": (163, 230, 53),
    "Gold": (251, 191, 36),
    "Paper White": (240, 240, 240),
}

st.set_page_config(page_title="GAN Digit Lab", page_icon="✍️", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1b2a5c55 0%, transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, #6d28d933 0%, transparent 55%),
                    #0b1020;
    }
    .block-container { padding-top: 3.5rem; max-width: 1150px; }
    .hero-title {
        font-size: clamp(2.1rem, 5vw, 3.4rem); font-weight: 800; line-height: 1.05;
        background: linear-gradient(90deg, #22d3ee, #a78bfa 55%, #f472b6);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        display: inline-block;
    }
    .hero { margin-bottom: .3rem; }
    .hero-emoji { font-size: clamp(1.8rem, 4vw, 2.8rem); margin-left: .5rem; vertical-align: middle; }
    .hero-sub { color: #aab3d0; font-size: 1.1rem; margin-bottom: 1.2rem; }
    .pill {
        display: inline-block; padding: .22rem .75rem; margin: 0 .35rem .35rem 0;
        border: 1px solid #2f3a66; border-radius: 999px; color: #c7d0f0;
        background: #131a33; font-size: .82rem;
    }
    [data-testid="stImage"] img {
        border-radius: 14px; border: 1px solid #2f3a66;
        box-shadow: 0 0 28px #22d3ee22;
    }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
    .stTabs [data-baseweb="tab"] {
        background: #131a33; border-radius: 10px; padding: .45rem 1rem; color: #c7d0f0;
    }
    .stTabs [aria-selected="true"] { background: #22d3ee22; color: #22d3ee; }
    .footer { color: #6f7aa3; font-size: .85rem; text-align: center; margin-top: 2.5rem; }
    .footer a { color: #8aa0ff; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the generator...")
def get_generator():
    return gan.load_generator(MODEL_PATH)


def render(images: np.ndarray, ncols: int, color, scale: int) -> Image.Image:
    """(n, 28, 28) uint8 -> coloured, enlarged PIL image."""
    grid = gan.tile_images(images, ncols=ncols)
    rgb = (grid[..., None].astype(np.float32) / 255.0 * np.array(color, dtype=np.float32)).astype(np.uint8)
    img = Image.fromarray(rgb)
    return img.resize((img.width * scale, img.height * scale), Image.BICUBIC)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def new_style():
    st.session_state.seed = random.randint(0, 999_999)


st.session_state.setdefault("seed", 42)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="hero"><span class="hero-title">GAN Digit Lab</span>'
    '<span class="hero-emoji">✍️</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Handwriting that <b>no human wrote</b>. '
    "A neural network invents every digit from pure random noise.</div>"
    '<span class="pill">PyTorch</span><span class="pill">Conditional GAN</span>'
    '<span class="pill">MNIST</span><span class="pill">Streamlit</span>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Model check
# --------------------------------------------------------------------------- #
if not MODEL_PATH.exists():
    st.warning("No trained model found yet.", icon="⚠️")
    st.markdown(
        """
        **Train it once (a few minutes on a free Google Colab GPU):**

        1. Open `train_colab.ipynb` in Google Colab and set the runtime to **GPU**.
        2. Run all cells. They call `train_gan(epochs=30)` and download `generator.pt`.
        3. Put the file at `models/generator.pt` and refresh this page.
        """
    )
    st.stop()

generator = get_generator()
n_params = sum(p.numel() for p in generator.parameters())
ckpt_epochs = getattr(generator, "trained_epochs", "?")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🎛️ Controls")
    ink = st.radio("Ink colour", list(INK_COLORS), index=0)
    color = INK_COLORS[ink]
    st.button("🎲 New handwriting style", on_click=new_style)
    st.caption(f"Style seed: `{st.session_state.seed}`  (same seed = same handwriting)")
    st.divider()
    st.subheader("Model card")
    st.markdown(
        f"- Type: conditional GAN\n- Parameters: **{n_params / 1e6:.2f} M**\n"
        f"- Trained for: **{ckpt_epochs} epochs**\n- Data: MNIST (60,000 digits)"
    )

seed = st.session_state.seed

tab_write, tab_gallery, tab_morph, tab_about = st.tabs(
    ["✍️ Write a number", "🖼️ Digit gallery", "🌀 Style morph", "🧠 How it works"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - Write a number
# --------------------------------------------------------------------------- #
with tab_write:
    st.subheader("Type a number, get AI handwriting")
    text = st.text_input("Digits (0-9)", value="2026", max_chars=12, key="number_text")
    digits = [int(c) for c in text if c.isdigit()]
    if not digits:
        st.info("Type at least one digit, for example **1234**.")
    else:
        imgs = gan.generate(
            generator,
            gan.digits_to_onehot(digits),
            gan.make_noise(len(digits), generator.latent_dim, seed),
        )
        strip = render(imgs, ncols=len(digits), color=color, scale=6)
        st.image(strip)
        st.download_button("⬇️ Download PNG", png_bytes(strip), "handwriting.png", "image/png")
        st.caption("Tip: press **New handwriting style** in the sidebar - every click is a new 'writer'.")

# --------------------------------------------------------------------------- #
# Tab 2 - Gallery
# --------------------------------------------------------------------------- #
with tab_gallery:
    st.subheader("One digit, endless styles")
    c1, c2 = st.columns([1, 2])
    digit = c1.selectbox("Digit", list(range(10)), index=7)
    count = c2.slider("How many variations", 8, 40, 24, step=8)
    imgs = gan.generate(
        generator,
        gan.digits_to_onehot([digit] * count),
        gan.make_noise(count, generator.latent_dim, seed),
    )
    grid = render(imgs, ncols=8, color=color, scale=4)
    st.image(grid)
    st.download_button("⬇️ Download PNG", png_bytes(grid), f"digit_{digit}.png", "image/png", key="dl_gallery")

# --------------------------------------------------------------------------- #
# Tab 3 - Style morph (walk through the latent space)
# --------------------------------------------------------------------------- #
def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation between two noise vectors (keeps the noise 'size' stable)."""
    a_n, b_n = a / a.norm(), b / b.norm()
    omega = torch.acos((a_n * b_n).sum().clamp(-1 + 1e-6, 1 - 1e-6))
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


with tab_morph:
    st.subheader("Glide between two handwriting styles")
    c1, c2 = st.columns([1, 2])
    m_digit = c1.selectbox("Digit", list(range(10)), index=3, key="morph_digit")
    steps = c2.slider("Steps", 4, 12, 8)
    z_pair = gan.make_noise(2, generator.latent_dim, seed)
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    z = slerp(z_pair[0], z_pair[1], t)
    frames = gan.generate(generator, gan.digits_to_onehot([m_digit] * steps), z)
    st.image(render(frames, ncols=steps, color=color, scale=5))
    st.caption(
        "Two random noise vectors are blended step by step. Every frame in between is a handwriting "
        "that never existed - this is the GAN's 'latent space' at work."
    )

# --------------------------------------------------------------------------- #
# Tab 4 - How it works
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("How does it work?")
    st.markdown(
        """
        A **GAN** is a game between two neural networks:

        - 🎨 **Generator** - the forger. It gets random noise plus a digit label and paints a 28x28 image.
        - 🕵️ **Discriminator** - the detective. It looks at an image and says *real* or *fake*.

        They train against each other. The detective gets better at spotting fakes, so the forger
        has to get better at faking, until its digits look hand-written.

        This is a **conditional** GAN: the digit label is fed to both networks, so you can *ask* for
        a specific digit. The random noise decides the handwriting style.
        """
    )
    progress, samples = ASSETS / "training_progress.png", ASSETS / "samples.png"
    if progress.exists():
        st.markdown("**The generator learning, epoch by epoch** (same noise every row)")
        st.image(str(progress))
    if samples.exists():
        st.markdown("**Random samples after training**")
        st.image(str(samples))

st.markdown(
    '<div class="footer">Built with PyTorch and Streamlit by Rahat Mia · '
    '<a href="https://github.com/rahat-ai-dev" target="_blank">github.com/rahat-ai-dev</a></div>',
    unsafe_allow_html=True,
)
