# Image Colorization (U-Net & pix2pix GAN)

Two approaches to colorizing grayscale images. Both work in LAB color space — the model takes the lightness channel (`L`) and predicts the two color channels (`ab`), so it only has to learn color, not brightness.

- **`Colorization_Unet_Skip_Connection.ipynb`** — a U-Net with skip connections trained with MSE. Fast and stable, but colors come out a bit muted.
- **`Colorization_GAN.ipynb`** — a pix2pix GAN using the same U-Net as generator with a PatchGAN discriminator, trained with L1 + adversarial loss. Slower, but gives more realistic, saturated colors.

RGB↔LAB conversion is done on the GPU with `kornia`, and training uses mixed precision (`torch.amp`). Trained on a Colab T4 with the [Google Universal Image Embeddings 128×128](https://www.kaggle.com/datasets/rhtsingh/google-universal-image-embeddings-128x128) dataset.

## Setup

```bash
pip install torch torchvision kornia kagglehub tqdm matplotlib pillow
```

## Pretrained weights

Weights are on Google Drive (too big for GitHub):

- U-Net (MSE): [download](<https://drive.google.com/file/d/1wiRCWvc8ULv98Hj9cvdn8CXAqwB-1LBj/view?usp=sharing>)
- pix2pix generator: [download](<https://drive.google.com/file/d/1jMIZwwHVoGPM17fV7jOvoQyd_xZdxx65/view?usp=sharing>)

## Inference

```python
import torch, kornia
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)          # or UNet_Generator() for the GAN
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

img = Image.open("input.jpg").convert("RGB")
rgb = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])(img).unsqueeze(0).to(device)

with torch.no_grad():
    lab = kornia.color.rgb_to_lab(rgb)
    L = lab[:, 0:1] / 100.0
    pred_ab = model(L)
    out_lab = torch.cat([L * 100.0, pred_ab * 128.0], dim=1)
    out_rgb = kornia.color.lab_to_rgb(out_lab).clamp(0, 1)
```

## Note

Colorization is ambiguous — a gray car could be any color — so the goal is plausible color, not exact recovery of the original.
