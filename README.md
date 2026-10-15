# GAN Digit Lab ✍️

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://gan-digit-generator-knkck5ybjlntvwyw56ycwz.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-gan--digit--generator-181717?logo=github)](https://github.com/rahat-ai-dev/gan-digit-generator)

**🚀 Live demo:** https://gan-digit-generator-knkck5ybjlntvwyw56ycwz.streamlit.app/  
**📦 Repository:** https://github.com/rahat-ai-dev/gan-digit-generator

**Handwriting that no human wrote.** A conditional GAN (built with PyTorch) learns from MNIST and invents brand-new handwritten digits from pure random noise. A Streamlit app lets you type any number and watch the AI "write" it.

## Features

- ✍️ **Write a number** - type `2026` and get AI handwriting for every digit, with downloadable PNG.
- 🖼️ **Digit gallery** - one digit, dozens of different handwriting styles.
- 🌀 **Style morph** - glide smoothly between two handwriting styles and see the GAN's latent space at work.
- 🎨 **Ink colours and a "new handwriting style" button** - every click is a new virtual writer.
- 🧠 **How it works** tab with a training-progress picture (the generator improving epoch by epoch).

## How it works

A GAN is a game between two neural networks:

| Network | Role | Input → Output |
|---|---|---|
| **Generator** | the forger | random noise (100 numbers) + digit label → 28×28 image |
| **Discriminator** | the detective | image + digit label → real or fake? |

Each training step has two parts:

1. **Train the Discriminator** on real MNIST digits (label 1) and the Generator's fakes (label 0).
2. **Train the Generator** to make fakes that the Discriminator calls *real*.

This is a **conditional** GAN: the digit label goes into both networks, so you can ask for a specific digit. The noise vector controls the handwriting *style*. Same noise = same style, which is why the seed in the app reproduces the same handwriting.

## Project structure

```
gan-digit-generator/
├── app.py               # Streamlit UI
├── gan.py               # models + train_gan() + load_generator() + generate()
├── train_colab.ipynb    # one-click training notebook for Google Colab
├── requirements.txt
├── models/              # generator.pt goes here (created by training)
├── assets/              # preview images (created by training)
└── .streamlit/config.toml
```

## Quick start

### 1. Train the model (Google Colab, free GPU)

1. Open `train_colab.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook, or open it from GitHub).
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The key line is:

   ```python
   from gan import train_gan
   train_gan(epochs=100)
   ```
4. Download `generator.pt` and put it in the `models/` folder (and the two PNGs into `assets/`).

### 2. Run the app locally

```bash
git clone https://github.com/rahat-ai-dev/gan-digit-generator.git
cd gan-digit-generator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Train on your own machine instead

```bash
python -c "from gan import train_gan; train_gan(epochs=100)"
```
(Much faster with a GPU. On CPU, lower the epochs.)

## Deploy for free (Streamlit Community Cloud)

1. Commit `models/generator.pt` (about 3 MB) and `assets/*.png` to your GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), choose the repo, and set the main file to `app.py`.

## Tech stack

Python · PyTorch · Streamlit · NumPy · Pillow

## What I learned

- How the Generator and Discriminator train against each other, and why `detach()` matters.
- Conditional generation: controlling *what* the GAN draws with a label and *how* it draws with noise.
- Stabilising GAN training: Adam with `betas=(0.5, 0.999)`, label smoothing, BatchNorm.
- Saving a model in Colab and serving it in a lightweight web app.

## Ideas to extend

- Train on Fashion-MNIST or EMNIST letters.
- Replace the generator with a deeper DCGAN or a Wasserstein GAN.
- Add a "draw a digit" canvas and compare it with the GAN output.

## Author

Built by **Rahat Mia** - [github.com/rahat-ai-dev](https://github.com/rahat-ai-dev)

Try it live: [GAN Digit Lab on Streamlit](https://gan-digit-generator-knkck5ybjlntvwyw56ycwz.streamlit.app/)

If you like the project, give it a ⭐