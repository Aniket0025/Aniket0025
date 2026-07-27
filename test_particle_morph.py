import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from build_banner import process_portrait, generate_logos, PORTRAIT_GRID_W, PORTRAIT_GRID_H

# 1. Sample 900 points from portrait dots
dots = process_portrait('dark')
idx = np.random.choice(len(dots), size=900, replace=False)
p_photo = dots[idx].astype(float)

# 2. Get logos
p1_raw, p2_raw, p3_raw = generate_logos(900)

# Optimal transport chain: p_photo -> p1 -> p2 -> p3 -> p_photo
# Match p_photo -> p1
cost0 = np.linalg.norm(p_photo[:, None, :] - p1_raw[None, :, :], axis=2)
_, c0 = linear_sum_assignment(cost0)
p1 = p1_raw[c0]

# Match p1 -> p2
cost1 = np.linalg.norm(p1[:, None, :] - p2_raw[None, :, :], axis=2)
_, c1 = linear_sum_assignment(cost1)
p2 = p2_raw[c1]

# Match p2 -> p3
cost2 = np.linalg.norm(p2[:, None, :] - p3_raw[None, :, :], axis=2)
_, c2 = linear_sum_assignment(cost2)
p3 = p3_raw[c2]

print("Photo particles shape:", p_photo.shape)
print("p1 (Flutter) matched:", p1.shape)
print("p2 (Code) matched:", p2.shape)
print("p3 (Vercel) matched:", p3.shape)
print("Chain matching complete!")
