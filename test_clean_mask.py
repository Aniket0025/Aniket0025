import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from scipy.ndimage import binary_fill_holes, binary_closing

img = Image.open('photo.png').convert('RGB')
target_w, target_h = 300, 340
img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
img_np = np.array(img_resized)

bg_sample = np.mean(img_np[:25, :25, :3], axis=(0,1))
dist = np.linalg.norm(img_np[:, :, :3].astype(float) - bg_sample, axis=2)

mask = dist > 30.0

# Clip out bottom corner bushes (x < 50 or x > 250 when y > 250)
Y, X = np.ogrid[:target_h, :target_w]
bush_left = (Y > 250) & (X < 45)
bush_right = (Y > 240) & (X > 252)
mask[bush_left] = False
mask[bush_right] = False

mask = binary_closing(mask, structure=np.ones((5,5)))
mask = binary_fill_holes(mask)

# Save mask preview to check
mask_img = Image.fromarray((mask * 255).astype(np.uint8))
mask_img.save('mask_preview.png')
print("Mask cleaned up successfully!")
