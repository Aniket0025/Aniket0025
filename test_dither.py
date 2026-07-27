import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import binary_fill_holes, binary_closing

img = Image.open('photo.png').convert('RGB')
target_w, target_h = 300, 340

img_cropped = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
img_np = np.array(img_cropped)

# Background sample from top corners
bg_sample = np.mean(img_np[:20, :20, :3], axis=(0,1))
dist = np.linalg.norm(img_np[:, :, :3].astype(float) - bg_sample, axis=2)
mask = dist > 35.0
mask = binary_closing(mask, structure=np.ones((5,5)))
mask = binary_fill_holes(mask)

gray = img_cropped.convert('L')
gray = ImageEnhance.Contrast(gray).enhance(1.3)
gray = ImageOps.autocontrast(gray, cutoff=1)
gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))

gray_arr = np.array(gray, dtype=float)

dither = gray_arr.copy()
out_bits = np.zeros_like(dither, dtype=int)

for y in range(target_h):
    if y % 2 == 0:
        for x in range(target_w):
            old_val = dither[y, x]
            new_val = 255 if old_val > 128 else 0
            out_bits[y, x] = new_val
            err = old_val - new_val
            if x + 1 < target_w:
                dither[y, x + 1] += err * 7 / 16.0
            if y + 1 < target_h:
                if x - 1 >= 0:
                    dither[y + 1, x - 1] += err * 3 / 16.0
                dither[y + 1, x] += err * 5 / 16.0
                if x + 1 < target_w:
                    dither[y + 1, x + 1] += err * 1 / 16.0
    else:
        for x in range(target_w - 1, -1, -1):
            old_val = dither[y, x]
            new_val = 255 if old_val > 128 else 0
            out_bits[y, x] = new_val
            err = old_val - new_val
            if x - 1 >= 0:
                dither[y, x - 1] += err * 7 / 16.0
            if y + 1 < target_h:
                if x + 1 < target_w:
                    dither[y + 1, x + 1] += err * 3 / 16.0
                dither[y + 1, x] += err * 5 / 16.0
                if x - 1 >= 0:
                    dither[y + 1, x - 1] += err * 1 / 16.0

dark_dots = (out_bits == 255) & mask
light_dots = (out_bits == 0)

print(f"Dark mode dot count: {np.sum(dark_dots)}")
print(f"Light mode dot count: {np.sum(light_dots)}")
