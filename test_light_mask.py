from build_banner import *

img = Image.open('photo.png').convert('RGB')
img_resized = img.resize((PORTRAIT_GRID_W, PORTRAIT_GRID_H), Image.Resampling.LANCZOS)
img_np = np.array(img_resized)

bg_sample = np.mean(img_np[:25, :25, :3], axis=(0,1))
dist = np.linalg.norm(img_np[:, :, :3].astype(float) - bg_sample, axis=2)

mask = dist > 30.0
mask = binary_closing(mask, structure=np.ones((5,5)))
mask = binary_fill_holes(mask)

gray = img_resized.convert('L')
gray = ImageEnhance.Contrast(gray).enhance(1.3)
gray = ImageOps.autocontrast(gray, cutoff=1)
gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))

gray_arr = np.array(gray, dtype=float)

dither = gray_arr.copy()
out_bits = np.zeros_like(dither, dtype=int)

for y in range(PORTRAIT_GRID_H):
    if y % 2 == 0:
        for x in range(PORTRAIT_GRID_W):
            old_val = dither[y, x]
            new_val = 255 if old_val > 128 else 0
            out_bits[y, x] = new_val
            err = old_val - new_val
            if x + 1 < PORTRAIT_GRID_W:
                dither[y, x + 1] += err * 7 / 16.0
            if y + 1 < PORTRAIT_GRID_H:
                if x - 1 >= 0:
                    dither[y + 1, x - 1] += err * 3 / 16.0
                dither[y + 1, x] += err * 5 / 16.0
                if x + 1 < PORTRAIT_GRID_W:
                    dither[y + 1, x + 1] += err * 1 / 16.0
    else:
        for x in range(PORTRAIT_GRID_W - 1, -1, -1):
            old_val = dither[y, x]
            new_val = 255 if old_val > 128 else 0
            out_bits[y, x] = new_val
            err = old_val - new_val
            if x - 1 >= 0:
                dither[y, x - 1] += err * 7 / 16.0
            if y + 1 < PORTRAIT_GRID_H:
                if x + 1 < PORTRAIT_GRID_W:
                    dither[y + 1, x + 1] += err * 3 / 16.0
                dither[y + 1, x] += err * 5 / 16.0
                if x - 1 >= 0:
                    dither[y + 1, x - 1] += err * 1 / 16.0

dot_mask = (out_bits == 0) & mask
ys, xs = np.where(dot_mask)
print("Light mode subject dot count:", len(ys))
