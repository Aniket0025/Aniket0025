import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

def get_logo_points(num_points=900, logo_type="vercel"):
    img = Image.new('L', (300, 340), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = 150, 170
    
    if logo_type == "vercel":
        # Draw large filled triangle
        pts = [(cx, cy - 90), (cx - 100, cy + 80), (cx + 100, cy + 80)]
        draw.polygon(pts, fill=255)
    elif logo_type == "code":
        # Draw </ > text or paths
        # <
        draw.line([(cx - 30, cy - 60), (cx - 90, cy), (cx - 30, cy + 60)], fill=255, width=22)
        # /
        draw.line([(cx + 25, cy - 70), (cx - 25, cy + 70)], fill=255, width=20)
        # >
        draw.line([(cx + 30, cy - 60), (cx + 90, cy), (cx + 30, cy + 60)], fill=255, width=22)
    elif logo_type == "flutter":
        # Flutter logo (two parallelograms/chevrons)
        pts1 = [(cx - 20, cy - 90), (cx + 70, cy - 90), (cx - 10, cy - 10), (cx - 100, cy - 10)]
        draw.polygon(pts1, fill=255)
        pts2 = [(cx - 30, cy + 10), (cx + 60, cy + 10), (cx - 20, cy + 90), (cx - 110, cy + 90)]
        draw.polygon(pts2, fill=255)
        
    arr = np.array(img)
    ys, xs = np.where(arr > 128)
    
    # Subsample or supersample to exact num_points
    idx = np.random.choice(len(xs), size=num_points, replace=(len(xs) < num_points))
    pts = np.column_stack([xs[idx], ys[idx]]).astype(float)
    return pts

# Test optimal transport matching between 3 logos
p1 = get_logo_points(900, "flutter")
p2 = get_logo_points(900, "code")
p3 = get_logo_points(900, "vercel")

# Match p1 -> p2
cost_matrix1 = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
row_ind1, col_ind1 = linear_sum_assignment(cost_matrix1)
p2_ordered = p2[col_ind1]

# Match p2 -> p3
cost_matrix2 = np.linalg.norm(p2_ordered[:, None, :] - p3[None, :, :], axis=2)
row_ind2, col_ind2 = linear_sum_assignment(cost_matrix2)
p3_ordered = p3[col_ind2]

print("P1 shape:", p1.shape)
print("P2 matched shape:", p2_ordered.shape)
print("P3 matched shape:", p3_ordered.shape)
print("Optimal transport matching succeeded!")
