import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.models import inception_v3
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.linalg import sqrtm
# Use TkAgg for matplotlib backend
matplotlib.use('TkAgg')

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DIP model (a simple CNN)
class DIP(nn.Module):
    def __init__(self):
        super(DIP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define DDPM model
class DDPM(nn.Module):
    def __init__(self):
        super(DDPM, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def initialize_with(self, dip_output):
        self.initial_state = dip_output

# Load CIFAR-10 dataset
def load_cifar10(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Training function for DIP
def train_dip(model, target_image, epochs=1000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(target_image)
        loss = criterion(output, target_image)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    return model(target_image).detach()

# Training function for DDPM
def train_ddpm(model, initial_state, num_steps=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(initial_state)
        loss = criterion(output, initial_state)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f'Step [{step}/{num_steps}], Loss: {loss.item():.4f}')
    return model

# Evaluate generated images
def evaluate(generated_images, save_path='generated_image.png'):
    save_image(generated_images, save_path)

# Calculate PSNR
def calculate_psnr(image1, image2):
    batch_size = image1.shape[0]
    psnr_values = []
    for i in range(batch_size):
        img1 = image1[i].detach().permute(1, 2, 0).cpu().numpy()
        img2 = image2[i].detach().permute(1, 2, 0).cpu().numpy()
        psnr_values.append(peak_signal_noise_ratio(img1, img2))
    return np.mean(psnr_values)

# Calculate SSIM
def calculate_ssim(image1, image2, win_size=3):
    batch_size = image1.shape[0]
    ssim_values = []
    data_range = 1.0  # Assuming images are normalized to [0, 1] range
    for i in range(batch_size):
        img1 = image1[i].detach().permute(1, 2, 0).cpu().numpy()
        img2 = image2[i].detach().permute(1, 2, 0).cpu().numpy()
        ssim_values.append(structural_similarity(img1, img2, multichannel=True, win_size=win_size, data_range=data_range))
    return np.mean(ssim_values)

# Calculate FID
def calculate_fid(images1, images2):
    inception = inception_v3(pretrained=True, transform_input=False).eval().cuda().to(device)
    def get_activations(images):
        images = images.to(device)  # Move input tensor to GPU
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        return inception(images).detach().cpu().numpy()

    act1 = get_activations(images1)
    act2 = get_activations(images2)
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # Check for singular covariance matrix
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    fid_value = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_value

# Evaluate the model
def evaluate_model(model, target_image, initial_state):
    generated_images = model(initial_state)
    psnr = calculate_psnr(generated_images, target_image)
    ssim = calculate_ssim(generated_images, target_image)
    return generated_images, psnr, ssim

# Load a batch of images for FID calculation
cifar10_loader = load_cifar10(batch_size=10)
real_images, _ = next(iter(cifar10_loader))

# Initialize and train the DIP model with various durations
dip_epochs_list = [500, 1000, 1500]
results_dip = []
for dip_epochs in dip_epochs_list:
    dip_model = DIP().to(device)
    target_image, _ = next(iter(cifar10_loader))
    target_image = target_image.to(device)
    dip_output = train_dip(dip_model, target_image, epochs=dip_epochs)
    ddpm_model = DDPM().to(device)
    ddpm_model.initialize_with(dip_output)
    ddpm_model = train_ddpm(ddpm_model, ddpm_model.initial_state, num_steps=500)
    generated_images_with_dip, psnr_with_dip, ssim_with_dip = evaluate_model(ddpm_model, target_image, dip_output)
    fid_with_dip = calculate_fid(generated_images_with_dip, real_images)
    results_dip.append((dip_epochs, psnr_with_dip, ssim_with_dip, fid_with_dip))
    # Save evaluation images
    evaluate(generated_images_with_dip, save_path=f'generated_with_dip_{dip_epochs}_epochs.png')

# Initialize and train the DDPM model with various noise schedules
num_steps_list = [300, 500, 1000]
results_ddpm = []
for num_steps in num_steps_list:
    dip_model = DIP().to(device)
    dip_output = train_dip(dip_model, target_image, epochs=1000)
    ddpm_model = DDPM().to(device)
    ddpm_model.initialize_with(dip_output)
    ddpm_model = train_ddpm(ddpm_model, ddpm_model.initial_state, num_steps=num_steps)
    generated_images_with_dip, psnr_with_dip, ssim_with_dip = evaluate_model(ddpm_model, target_image, dip_output)
    fid_with_dip = calculate_fid(generated_images_with_dip, real_images)
    results_ddpm.append((num_steps, psnr_with_dip, ssim_with_dip, fid_with_dip))
    # Save evaluation images
    evaluate(generated_images_with_dip, save_path=f'generated_with_dip_{num_steps}_steps.png')

# Save and plot results
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

def plot_results(results, xlabel, title):
    metrics = ['PSNR', 'SSIM', 'FID']
    for i, metric in enumerate(metrics):
        values = [result[i+1] for result in results]
        labels = [result[0] for result in results]
        plt.figure(figsize=(8, 6))
        plt.plot(labels, values, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(metric)
        plt.title(f'{title} vs {metric}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{title}_{metric}.png'))
        plt.show()

plot_results(results_dip, 'DIP Training Epochs', 'DIP Training Epochs')
plot_results(results_ddpm, 'DDPM Noise Steps', 'DDPM Noise Steps')

# Save printed results to a text file
with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
    f.write('Results for varying DIP training epochs:\n')
    f.write('DIP Training Epochs | PSNR | SSIM | FID\n')
    for result in results_dip:
        f.write(f'{result[0]} | {result[1]:.4f} | {result[2]:.4f} | {result[3]:.4f}\n')

    f.write('\nResults for varying DDPM noise steps:\n')
    f.write('DDPM Noise Steps | PSNR | SSIM | FID\n')
    for result in results_ddpm:
        f.write(f'{result[0]} | {result[1]:.4f} | {result[2]:.4f} | {result[3]:.4f}\n')
