# Image Colorization with Deep Learning

Two models that take a black and white image and predict its colors. One is a plain U-Net trained with MSE loss, the other wraps the same U-Net in a Pix2Pix style GAN. Both work on 128x128 images.

## The idea

A grayscale image only carries brightness, no color. Instead of predicting full RGB, the models work in **LAB color space**:

- **L** is the lightness. This is basically the grayscale image, and it's used as the input.
- **A** and **B** hold the color. This is what the model has to predict.

So the network gets the L channel and learns to output the A and B channels, which are then combined back into a full color image. The RGB to LAB conversion happens on the GPU during training using `kornia`.

## Two approaches

### 1. U-Net (`Colorization_Unet.ipynb`)

A standard encoder-decoder U-Net with skip connections. The encoder goes `1 → 64 → 128 → 256 → 512` channels, with a 1024 channel bottleneck in the middle, then the decoder mirrors it back up to a final 1x1 conv that outputs the 2 color channels. It's trained to match the real AB channels directly.

- Loss: MSE
- Optimizer: Adam, lr `1e-4`
- 20 epochs, batch size 128, mixed precision

### 2. GAN (`Colorization_with_GAN.ipynb`)

The same U-Net is used as the generator, plus a PatchGAN discriminator that looks at `(L, AB)` pairs and decides whether they look real or generated. This is the Pix2Pix setup.

- Generator loss: adversarial (BCE) + `100 × L1`
- Discriminator loss: BCE
- Optimizer: Adam, lr `2e-4`, betas `(0.5, 0.999)`
- 9 epochs, batch size 64, mixed precision

## Dataset

[Google Universal Image Embeddings 128x128](https://www.kaggle.com/datasets/rhtsingh/google-universal-image-embeddings-128x128) (~130k images). It downloads automatically through `kagglehub` when you run the notebook. The data is split 90% train / 10% test.

## Results

The U-Net gives stable, natural looking colors, but they tend to come out a bit dull and washed out. That's expected with MSE, the model plays it safe and averages toward muted tones.

The GAN produces brighter, more convincing colors, but it's less predictable. It sometimes picks the wrong color or looks slightly off, which comes from the usual GAN training instability.

Short version: U-Net is safe but flat, GAN is vivid but riskier.

![GAN colorization results](results_gan.png)

*GAN output: grayscale input on top, colorized prediction below.*

## Running it

Both are notebooks, easiest to run on Colab or any machine with a GPU.

1. Open either notebook.
2. Run the cells top to bottom. The dataset downloads on its own.
3. Training saves the best weights:
   - U-Net → `best_colorization_model.pth`
   - GAN → `pix2pix_generator.pth`
4. The final cells show grayscale input, original, and colorized output side by side on a few test images.

### Requirements

`torch`, `torchvision`, `kornia`, `scikit-image`, `kagglehub`, `matplotlib`, `tqdm`

Colab already has most of these. `kornia` is installed in the first cell of each notebook.

## Files

- `Colorization_Unet.ipynb` — U-Net model
- `Colorization_with_GAN.ipynb` — Pix2Pix GANs — a gray car could be any color — so the goal is plausible color, not exact recovery of the original.
