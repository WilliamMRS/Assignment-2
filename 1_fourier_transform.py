import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr

img = data.camera().astype(float)
rows, cols = img.shape
centerrow, centercol = rows // 2, cols // 2

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))

# Task 1
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Frequency Spectrum')
plt.savefig('output/fourier/task1.png')
plt.show()



# Applying filter helping functions
Y, X = np.ogrid[:rows, :cols]
dist = np.sqrt((Y - centerrow) ** 2 + (X - centercol) ** 2)   # distance of each frequency from the centre
 
# 1 inside circle, 0 outside
def lowpass_mask(radius):
    return (dist <= radius).astype(float)
 
 # 0 inside circle, 1 outside
def highpass_mask(radius):
    return (dist > radius).astype(float)
 
# Apply mask
def apply_mask(fshift, mask):
    filtered = fshift * mask
    return np.fft.ifft2(np.fft.ifftshift(filtered)).real

# Task 2
rng = np.random.default_rng(0)
noisy = img + rng.normal(0, 20, img.shape) # add gaussian noise
noisy_fshift = np.fft.fftshift(np.fft.fft2(noisy))

plt.figure(figsize=(16, 8))
plt.subplot(2, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Original')

# Gaussian noise
plt.subplot(2, 4, 2), plt.imshow(noisy, cmap='gray'), plt.title(f'Original + Gaussian noise\nPSNR={psnr(img, noisy, data_range=255):.1f} dB')

# Filter mask r = 50
plt.subplot(2, 4, 3), plt.imshow(lowpass_mask(50), cmap='gray'), plt.title('Filter mask r=50')

# Low-pass filter r = 20
result_20 = apply_mask(noisy_fshift, lowpass_mask(20))
p_20 = psnr(img, np.clip(result_20, 0, 255), data_range=255)
plt.subplot(2, 4, 5)
plt.imshow(result_20, cmap='gray')
plt.title(f'Low-pass r=20\nPSNR={p_20:.1f} dB')
plt.axis('off')

# Low-pass filter r = 50
result_50 = apply_mask(noisy_fshift, lowpass_mask(50))
p_50 = psnr(img, np.clip(result_50, 0, 255), data_range=255)
plt.subplot(2, 4, 6)
plt.imshow(result_50, cmap='gray')
plt.title(f'Low-pass r=50\nPSNR={p_50:.1f} dB')
plt.axis('off')

# Low-pass filter r = 100
result_100 = apply_mask(noisy_fshift, lowpass_mask(100))
p_100 = psnr(img, np.clip(result_100, 0, 255), data_range=255)
plt.subplot(2, 4, 7)
plt.imshow(result_100, cmap='gray')
plt.title(f'Low-pass r=100\nPSNR={p_100:.1f} dB')
plt.axis('off')

plt.tight_layout()
plt.savefig('output/fourier/task2.png')
plt.show()



# Task 3
hp = apply_mask(fshift, highpass_mask(30))
 
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(highpass_mask(30), cmap='gray'), plt.title('High-pass mask')
plt.subplot(133), plt.imshow(np.abs(hp), cmap='gray'), plt.title('High-pass filter')
plt.tight_layout()
plt.savefig('output/fourier/task3.png')
plt.show()



# Task 4
percents = [100, 10, 1, 0.1]
 
plt.figure(figsize=(16, 8))
for i, p in enumerate(percents):
    threshold = np.percentile(np.abs(f), 100 - p)
    f_kept = f * (np.abs(f) >= threshold)
    recon = np.fft.ifft2(f_kept).real
    mse = np.mean((img - recon) ** 2)
    q = psnr(img, np.clip(recon, 0, 255), data_range=255)
    plt.subplot(2, 2, i + 1), plt.imshow(recon, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Keep {p}% (ratio {100 / p:.0f}:1)\nMSE={mse:.1f}, PSNR={q:.1f} dB')
    print(f'{p:>5}% kept | compression ~{100 / p:>5.0f}:1 | MSE {mse:8.2f} | PSNR {q:5.2f} dB')
plt.tight_layout()
plt.savefig('output/fourier/task4.png')
plt.show()