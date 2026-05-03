import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import io
import base64
import torch.nn as nn
from fastapi import UploadFile, File


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ===== Load Model =====
device = "cuda" if torch.cuda.is_available() else "cpu"


class forwardDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = "cpu"

    
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0) 

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,)).to(self.device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        epsilon = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    

diffusion = forwardDiffusion(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        time=time.float()
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, stride=1):
        super().__init__()

        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, out_ch)

        self.act = nn.SiLU()

    
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        # skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x, t):

        identity = self.skip(x)

        # --------- time embedding injection ---------
        t_emb = self.time_mlp(t)              # (B, out_ch)
        t_emb = t_emb[:, :, None, None]       # (B, out_ch, 1, 1)

        # --------- first block ---------
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # inject time here (after first conv like DDPM style)
        h = h + t_emb

        # --------- second block ---------
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        # residual connection
        return h + identity
    
class DiffusionUNet(nn.Module):

    def __init__(self, time_dim=256):
        super().__init__()

        self.time_dim = time_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
)

        # ---------------- Encoder ----------------
        self.enc1 = ResBlock(3, 64, time_dim)
        self.down1 = nn.Conv2d(64, 64, 4, 2, 1)

        self.enc2 = ResBlock(64, 128, time_dim)
        self.down2 = nn.Conv2d(128, 128, 4, 2, 1)

        self.enc3 = ResBlock(128, 256, time_dim)

        # ---------------- Bottleneck ----------------
        self.mid = ResBlock(256, 256, time_dim)

        # ---------------- Decoder ----------------
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec1 = ResBlock(256, 128, time_dim)  # concat -> 128+128

        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2 = ResBlock(128, 64, time_dim)   # concat -> 64+64

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):

        t = self.time_embedding(t)

        # ---------------- Encoder ----------------
        x1 = self.enc1(x, t)
        x2 = self.down1(x1)

        x3 = self.enc2(x2, t)
        x4 = self.down2(x3)

        x5 = self.enc3(x4, t)

        x_mid = self.mid(x5, t)

        # ------------------------- Decoder ----------------
        x = self.up1(x_mid)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x, t)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x, t)

        return self.out(x)
    
model = DiffusionUNet()  # must match training
state_dict = torch.load("diffusion_model.pth", map_location=device)

# Remove 'module.' prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()
app = FastAPI()

# ===== Utils =====
def tensor_to_image(tensor):
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def encode_image(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@torch.no_grad()
def sample(model, diffusion, image_size=64, batch_size=1, channels=3, device="cpu"):
    model.eval()

    x = torch.randn(batch_size, channels, image_size, image_size).to(device)

    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        predicted_noise = model(x, t_tensor)

        alpha = diffusion.alphas[t]
        alpha_hat = diffusion.alpha_hat[t]
        beta = diffusion.betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (
            1 / torch.sqrt(alpha)
        ) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

    return x

def image_to_tensor(image: Image.Image, size=64):
    image = image.convert("RGB").resize((size, size))
    img = np.array(image).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1)  # CHW
    img = img * 2 - 1  # [-1, 1]
    return img.unsqueeze(0).to(device)

@torch.no_grad()
def denoise_image(model, diffusion, x0, device="cpu"):
    b = x0.shape[0]

    # random timestep
    t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()

    noisy_x, true_noise = diffusion.noise_images(x0, t)

    predicted_noise = model(noisy_x, t)

    # reconstruct x0 (basic DDPM estimate)
    alpha_hat = diffusion.alpha_hat[t][:, None, None, None]

    reconstructed = (noisy_x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)

    return noisy_x, reconstructed

def compute_metrics(original, reconstructed):
    orig = tensor_to_image(original.squeeze(0))
    recon = tensor_to_image(reconstructed.squeeze(0))

    psnr_val = psnr(orig, recon, data_range=255)
    ssim_val = ssim(orig, recon, channel_axis=2, data_range=255)

    return psnr_val, ssim_val, orig, recon

@app.post("/generate")
def generate():
    with torch.no_grad():
        x = sample(model, diffusion, image_size=64, batch_size=1, device=device)

        img = tensor_to_image(x.squeeze(0))
        encoded = encode_image(img)

        return {
            "image": encoded
        }
    

@app.post("/denoise")
async def denoise(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read()))
    x0 = image_to_tensor(image)

    noisy, recon = denoise_image(model, diffusion, x0)

    psnr_val, ssim_val, orig, recon_img = compute_metrics(x0, recon)

    # convert noisy tensor to image
    noisy_img = tensor_to_image(noisy.squeeze(0))

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "original": encode_image(orig),
        "noisy": encode_image(noisy_img),
        "reconstructed": encode_image(recon_img)
    }