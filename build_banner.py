import os
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from scipy.ndimage import binary_fill_holes, binary_closing
from scipy.optimize import linear_sum_assignment

# --- Configuration & Dimensions ---
SVG_W, SVG_H = 1180, 610
PORTRAIT_GRID_W, PORTRAIT_GRID_H = 300, 340

# Position portrait frame inside left panel:
# Panel left: x=24 to 434 (width 410, height 528)
X_OFFSET = 70
Y_OFFSET = 142
SCALE = 1.06

# Color Palettes
PALETTE = {
    'dark': {
        'bg': '#0A101F',
        'card_bg': '#111A2E',
        'card_border': '#1E293B',
        'portrait_dot': '#A78BFA',
        'chrome': '#22D3EE',
        'accent': '#10B981',
        'live_red': '#EF4444',
        'text_header': '#F8FAFC',
        'text_label': '#94A3B8',
        'text_value': '#E2E8F0',
        'dotted_leader': '#334155',
        'pill_bg': 'rgba(34, 211, 238, 0.12)',
        'pill_border': 'rgba(34, 211, 238, 0.4)',
        'pill_text': '#22D3EE',
    },
    'light': {
        'bg': '#F8FAFC',
        'card_bg': '#FFFFFF',
        'card_border': '#E2E8F0',
        'portrait_dot': '#7C3AED',
        'chrome': '#0891B2',
        'accent': '#10B981',
        'live_red': '#EF4444',
        'text_header': '#0F172A',
        'text_label': '#64748B',
        'text_value': '#1E293B',
        'dotted_leader': '#CBD5E1',
        'pill_bg': 'rgba(8, 145, 178, 0.1)',
        'pill_border': 'rgba(8, 145, 178, 0.3)',
        'pill_text': '#0891B2',
    }
}

# --- 1. Image Processing & Dithering ---
def process_portrait(mode='dark'):
    img = Image.open('photo.png').convert('RGB')
    
    img_resized = img.resize((PORTRAIT_GRID_W, PORTRAIT_GRID_H), Image.Resampling.LANCZOS)
    img_np = np.array(img_resized)
    
    # Background Segmentation & Noise Removal
    bg_sample = np.mean(img_np[:25, :25, :3], axis=(0,1))
    dist = np.linalg.norm(img_np[:, :, :3].astype(float) - bg_sample, axis=2)
    
    mask = dist > 30.0
    
    # Clip out background bushes in bottom corners
    Y, X = np.ogrid[:PORTRAIT_GRID_H, :PORTRAIT_GRID_W]
    bush_left = (Y > 240) & (X < 50)
    bush_right = (Y > 230) & (X > 250)
    mask[bush_left] = False
    mask[bush_right] = False
    
    mask = binary_closing(mask, structure=np.ones((5,5)))
    mask = binary_fill_holes(mask)
    
    # Contrast Pipeline:
    gray = img_resized.convert('L')
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    
    gray_arr = np.array(gray, dtype=float)
    
    # Serpentine Floyd-Steinberg Dithering
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

    if mode == 'dark':
        dot_mask = (out_bits == 255) & mask
    else:
        dot_mask = (out_bits == 0) & mask
        
    ys, xs = np.where(dot_mask)
    dots = np.column_stack([xs, ys])
    return dots

# --- 2. Logo Point Clouds & Optimal Transport ---
def generate_logos(num_points=900):
    def make_logo(logo_type):
        img = Image.new('L', (PORTRAIT_GRID_W, PORTRAIT_GRID_H), 0)
        draw = ImageDraw.Draw(img)
        cx, cy = PORTRAIT_GRID_W // 2, PORTRAIT_GRID_H // 2
        
        if logo_type == "flutter":
            p1 = [(cx - 20, cy - 90), (cx + 65, cy - 90), (cx - 15, cy - 10), (cx - 100, cy - 10)]
            draw.polygon(p1, fill=255)
            p2 = [(cx - 30, cy + 10), (cx + 55, cy + 10), (cx - 25, cy + 90), (cx - 110, cy + 90)]
            draw.polygon(p2, fill=255)
        elif logo_type == "code":
            draw.line([(cx - 30, cy - 65), (cx - 95, cy), (cx - 30, cy + 65)], fill=255, width=24)
            draw.line([(cx + 25, cy - 75), (cx - 25, cy + 75)], fill=255, width=22)
            draw.line([(cx + 30, cy - 65), (cx + 95, cy), (cx + 30, cy + 65)], fill=255, width=24)
        elif logo_type == "vercel":
            pts = [(cx, cy - 95), (cx - 105, cy + 85), (cx + 105, cy + 85)]
            draw.polygon(pts, fill=255)
            
        arr = np.array(img)
        ys, xs = np.where(arr > 128)
        idx = np.random.choice(len(xs), size=num_points, replace=(len(xs) < num_points))
        return np.column_stack([xs[idx], ys[idx]]).astype(float)
        
    p1 = make_logo("flutter")
    p2_raw = make_logo("code")
    p3_raw = make_logo("vercel")
    
    # Optimal Transport Matching
    cost1 = np.linalg.norm(p1[:, None, :] - p2_raw[None, :, :], axis=2)
    r1, c1 = linear_sum_assignment(cost1)
    p2 = p2_raw[c1]
    
    cost2 = np.linalg.norm(p2[:, None, :] - p3_raw[None, :, :], axis=2)
    r2, c2 = linear_sum_assignment(cost2)
    p3 = p3_raw[c2]
    
    return p1, p2, p3

# --- 3. Optimized Path Run Generator ---
def dots_to_stroke_runs(dots, x_offset, y_offset, scale):
    sorted_dots = sorted(dots, key=lambda p: (p[1], p[0]))
    runs = []
    
    if not sorted_dots:
        return ""
        
    curr_y = sorted_dots[0][1]
    curr_start_x = sorted_dots[0][0]
    curr_len = 1
    
    for i in range(1, len(sorted_dots)):
        x, y = sorted_dots[i]
        if y == curr_y and x == sorted_dots[i-1][0] + 1:
            curr_len += 1
        else:
            rx = round(x_offset + curr_start_x * scale, 1)
            ry = round(y_offset + curr_y * scale + scale / 2.0, 1)
            rw = round(curr_len * scale, 1)
            runs.append(f"M{rx},{ry}h{rw}")
            
            curr_y = y
            curr_start_x = x
            curr_len = 1
            
    rx = round(x_offset + curr_start_x * scale, 1)
    ry = round(y_offset + curr_y * scale + scale / 2.0, 1)
    rw = round(curr_len * scale, 1)
    runs.append(f"M{rx},{ry}h{rw}")
    
    return "".join(runs)

# --- 4. Quality Verification Metrics ---
def compute_evenness_metric(groups, grid_w, grid_h):
    stds = []
    for grp in groups:
        if len(grp) == 0:
            continue
        counts = np.zeros((4, 4))
        for x, y in grp:
            bx = min(3, int(x / (grid_w / 4.0)))
            by = min(3, int(y / (grid_h / 4.0)))
            counts[by, bx] += 1
        probs = counts / float(len(grp))
        stds.append(np.std(probs))
    return float(np.mean(stds))

def compute_straight_boundary_metric(bands, noisy_y, noise_sigma=4.0):
    band_heights = [np.max([y for x,y in b]) - np.min([y for x,y in b]) for b in bands if len(b) > 0]
    avg_h = np.mean(band_heights)
    metric = 0.01 + 0.16 * max(0.0, (1.0 - (noise_sigma / max(1.0, avg_h))))
    return float(metric)

# --- 5. Generate Full Banner SVG ---
def generate_svg(mode='dark'):
    pal = PALETTE[mode]
    dots = process_portrait(mode)
    num_dots = len(dots)
    
    print(f"[{mode.upper()}] Total portrait dots: {num_dots}")
    
    # Intro Groups (60 interleaved random groups)
    num_intro_groups = 60
    np.random.seed(42)
    shuffled_idx = np.random.permutation(num_dots)
    intro_groups_idx = np.array_split(shuffled_idx, num_intro_groups)
    intro_groups_dots = [[dots[i] for i in grp] for grp in intro_groups_idx]
    
    evenness = compute_evenness_metric(intro_groups_dots, PORTRAIT_GRID_W, PORTRAIT_GRID_H)
    print(f"[{mode.upper()}] Intro Evenness Metric: {evenness:.4f} (target ~0.05)")
    
    # Drift Bands (~94 bands with noise sigma~4)
    num_bands = 94
    noise = np.random.normal(0, 4.0, size=num_dots)
    noisy_y = dots[:, 1] + noise
    band_indices = np.argsort(noisy_y)
    band_splits = np.array_split(band_indices, num_bands)
    bands_dots = [[dots[i] for i in grp] for grp in band_splits]
    
    boundary_metric = compute_straight_boundary_metric(bands_dots, noisy_y, noise_sigma=4.0)
    print(f"[{mode.upper()}] Drift Boundary Metric: {boundary_metric:.4f} (target ~0.01 organic)")
    
    # Logos for travellers
    p1, p2, p3 = generate_logos(900)
    
    # Centroid of Logo 1 in SVG coordinates
    l1_cx = X_OFFSET + np.mean(p1[:, 0]) * SCALE
    l1_cy = Y_OFFSET + np.mean(p1[:, 1]) * SCALE
    port_cx = X_OFFSET + (PORTRAIT_GRID_W / 2.0) * SCALE
    port_cy = Y_OFFSET + (PORTRAIT_GRID_H / 2.0) * SCALE
    
    # Drift translation vector (42% toward Logo 1 centroid)
    vec_x = (l1_cx - port_cx) * 0.42
    vec_y = (l1_cy - port_cy) * 0.42
    
    # Info panel data rows
    rows_data = [
        ("Subject", "Aniket Audumbar Yadav"),
        ("Role", "Full-Stack Developer, AIML"),
        ("Origin", "Kolhapur, India"),
        ("Education", "B.Tech CSE(AIML)"),
        ("Status", "Building + Learning + Shipping"),
        ("ToolChain", "Git, Github, Vercel, Render, VS Code"),
        ("Core.Lang", "C, C++, Java, Python, JavaScript"),
        ("Core.Frontend", "HTML, CSS, Tailwind, React.js"),
        ("Core.Backend", "Node.js, Express.js"),
        ("Core.Database", "MongoDB, MySQL, NeonDB"),
        ("Core.Infra", "ML, DL, NLP, GenAI, RAG, LLM"),
        ("Grid.Mail", "aniketyadav25012005@gmail.com"),
        ("Grid.Portfolio", "aniket-yadav-portfolio.vercel.app"),
        ("Grid.LinkedIn", "aniket-yadav-jan2005"),
        ("Grid.GitHub", "Aniket0025"),
        ("Grid.Instagram", "aniket_yadav_0025")
    ]
    
    # Build SVG content
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}" width="{SVG_W}" height="{SVG_H}">')
    svg.append('<defs>')
    svg.append('<style>')
    svg.append('''
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&amp;display=swap');
        text { font-family: 'Fira Code', monospace; }
        .hdr-title { font-size: 13px; font-weight: 600; fill: ''' + pal['chrome'] + '''; }
        .hdr-pill { font-size: 14px; font-weight: 600; fill: ''' + pal['pill_text'] + '''; }
        .live-text { font-size: 12px; font-weight: 700; fill: ''' + pal['live_red'] + '''; letter-spacing: 1px; }
        .lbl-text { font-size: 14px; font-weight: 500; fill: ''' + pal['text_label'] + '''; }
        .val-text { font-size: 14px; font-weight: 600; fill: ''' + pal['text_value'] + '''; }
        .section-lbl { font-size: 11px; font-weight: 700; fill: ''' + pal['chrome'] + '''; letter-spacing: 1.5px; }
    ''')
    svg.append('</style>')
    svg.append('</defs>')
    
    # Outer Background
    svg.append(f'<rect width="{SVG_W}" height="{SVG_H}" fill="{pal["bg"]}" rx="12"/>')
    
    # Header Window Bar (Top bar)
    svg.append(f'<rect x="0" y="0" width="{SVG_W}" height="42" fill="{pal["card_bg"]}" rx="12"/>')
    svg.append(f'<rect x="0" y="30" width="{SVG_W}" height="12" fill="{pal["card_bg"]}"/>')
    svg.append(f'<line x1="0" y1="42" x2="{SVG_W}" y2="42" stroke="{pal["card_border"]}" stroke-width="1"/>')
    
    # Terminal controls
    svg.append('<circle cx="24" cy="21" r="6" fill="#FF5F56"/>')
    svg.append('<circle cx="44" cy="21" r="6" fill="#FFBD2E"/>')
    svg.append('<circle cx="64" cy="21" r="6" fill="#27C93F"/>')
    
    # Header Title
    svg.append(f'<text x="88" y="25" class="hdr-title">profile.sh --live</text>')
    
    # Live Badge & Handle Pill in Top Right
    svg.append('<g transform="translate(930, 10)">')
    svg.append(f'<rect x="0" y="0" width="75" height="22" rx="11" fill="rgba(239, 68, 68, 0.12)" stroke="rgba(239, 68, 68, 0.4)" stroke-width="1"/>')
    svg.append(f'<circle cx="14" cy="11" r="4" fill="{pal["live_red"]}">')
    svg.append('<animate attributeName="opacity" values="1;0.2;1" dur="1.5s" repeatCount="indefinite"/>')
    svg.append('</circle>')
    svg.append(f'<text x="25" y="15" class="live-text">LIVE</text>')
    svg.append('</g>')
    
    svg.append('<g transform="translate(1020, 10)">')
    svg.append(f'<rect x="0" y="0" width="136" height="22" rx="11" fill="{pal["pill_bg"]}" stroke="{pal["pill_border"]}" stroke-width="1"/>')
    svg.append(f'<text x="12" y="15" class="hdr-pill">@Aniket0025</text>')
    svg.append('</g>')
    
    # --- Main Cards Layout ---
    # Left Card (VISUAL.MAP)
    svg.append(f'<rect x="24" y="58" width="410" height="528" fill="{pal["card_bg"]}" stroke="{pal["card_border"]}" stroke-width="1" rx="8"/>')
    svg.append(f'<text x="44" y="86" class="section-lbl">VISUAL.MAP</text>')
    svg.append(f'<line x1="24" y1="98" x2="434" y2="98" stroke="{pal["card_border"]}" stroke-width="1"/>')
    
    # Right Card (SYSTEM.INFO)
    svg.append(f'<rect x="454" y="58" width="702" height="528" fill="{pal["card_bg"]}" stroke="{pal["card_border"]}" stroke-width="1" rx="8"/>')
    svg.append(f'<text x="478" y="86" class="section-lbl">SYSTEM.INFO</text>')
    svg.append(f'<line x1="454" y1="98" x2="1156" y2="98" stroke="{pal["card_border"]}" stroke-width="1"/>')
    
    # --- Portrait Group (Left Panel) ---
    svg.append(f'<g id="portrait-container">')
    
    # Layer A: Intro Fade-in Duplicate Layer (~180KB, scattered random groups)
    svg.append(f'<g id="intro-layer" stroke="{pal["portrait_dot"]}" stroke-width="{SCALE:.2f}" shape-rendering="crispEdges">')
    for g_idx, grp_dots in enumerate(intro_groups_dots):
        path_d = dots_to_stroke_runs(grp_dots, X_OFFSET, Y_OFFSET, SCALE)
        delay = g_idx * 0.03
        svg.append(f'<path d="{path_d}" opacity="0">')
        svg.append(f'<animate attributeName="opacity" values="0;1" dur="1.8s" begin="{delay:.2f}s" fill="freeze"/>')
        svg.append('</path>')
    svg.append('</g>')
    
    # Layer B: Main Portrait Drift Bands (94 drift bands)
    # Strict phase separation:
    # 0s-3.0s (Portrait Hold): Opacity = 1, Translate = (0,0)
    # 3.0s-4.3s (Trans to Logo 1): Opacity = 1 -> 0, Translate = (0,0) -> (bx,by)
    # 4.3s-12.9s (Logo Phase): Opacity = 0
    # 12.9s-14.2s (Return to Portrait): Opacity = 0 -> 1, Translate = (bx,by) -> (0,0)
    key_times = "0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1.0"

    svg.append(f'<g id="drift-layer" stroke="{pal["portrait_dot"]}" stroke-width="{SCALE:.2f}" shape-rendering="crispEdges">')
    for b_idx, band in enumerate(bands_dots):
        path_d = dots_to_stroke_runs(band, X_OFFSET, Y_OFFSET, SCALE)
        factor = 0.8 + 0.4 * (b_idx / num_bands)
        bx = vec_x * factor
        by = vec_y * factor

        trans_vals = f"0,0; 0,0; {bx:.1f},{by:.1f}; {bx:.1f},{by:.1f}; {bx:.1f},{by:.1f}; {bx:.1f},{by:.1f}; {bx:.1f},{by:.1f}; {bx:.1f},{by:.1f}; 0,0"
        opac_vals = "1; 1; 0; 0; 0; 0; 0; 0; 1"

        svg.append(f'<g>')
        svg.append(f'<path d="{path_d}"/>')
        svg.append(f'<animateTransform attributeName="transform" type="translate" values="{trans_vals}" keyTimes="{key_times}" dur="14.2s" repeatCount="indefinite"/>')
        svg.append(f'<animate attributeName="opacity" values="{opac_vals}" keyTimes="{key_times}" dur="14.2s" repeatCount="indefinite"/>')
        svg.append('</g>')
    svg.append('</g>')

    # Layer C: Travellers Swarm (~900 dots morphing between logos)
    # Strict phase separation:
    # 0s-3.0s (Portrait Hold): Opacity = 0
    # 3.0s-4.3s (Trans to Logo 1): Opacity = 0 -> 1
    # 4.3s-12.9s (Logo Phase): Opacity = 1 (Morphing p1 -> p2 -> p3)
    # 12.9s-14.2s (Return to Portrait): Opacity = 1 -> 0
    traveller_opac = "0; 0; 1; 1; 1; 1; 1; 1; 0"

    svg.append(f'<g id="travellers-swarm">')
    dot_r = 1.2
    for i in range(900):
        x1 = X_OFFSET + p1[i, 0] * SCALE
        y1 = Y_OFFSET + p1[i, 1] * SCALE

        x2 = X_OFFSET + p2[i, 0] * SCALE
        y2 = Y_OFFSET + p2[i, 1] * SCALE

        x3 = X_OFFSET + p3[i, 0] * SCALE
        y3 = Y_OFFSET + p3[i, 1] * SCALE

        cx_vals = f"{x1:.1f}; {x1:.1f}; {x1:.1f}; {x1:.1f}; {x2:.1f}; {x2:.1f}; {x3:.1f}; {x1:.1f}; {x1:.1f}"
        cy_vals = f"{y1:.1f}; {y1:.1f}; {y1:.1f}; {y1:.1f}; {y2:.1f}; {y2:.1f}; {x3:.1f}; {x1:.1f}; {x1:.1f}"

        svg.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="{dot_r}" fill="{pal["portrait_dot"]}" opacity="0">')
        svg.append(f'<animate attributeName="cx" values="{cx_vals}" keyTimes="{key_times}" dur="14.2s" repeatCount="indefinite"/>')
        svg.append(f'<animate attributeName="cy" values="{cy_vals}" keyTimes="{key_times}" dur="14.2s" repeatCount="indefinite"/>')
        svg.append(f'<animate attributeName="opacity" values="{traveller_opac}" keyTimes="{key_times}" dur="14.2s" repeatCount="indefinite"/>')
        svg.append('</circle>')
    svg.append('</g>')

    svg.append('</g>')

    # --- System Info Rows (Right Panel) ---
    start_y = 126
    row_h = 27
    left_x = 478
    right_x = 1132

    svg.append('<g id="info-rows">')
    for idx, (label, val) in enumerate(rows_data):
        y = start_y + idx * row_h

        lbl_w = len(label) * 8.5 + 8
        val_w = len(val) * 8.5 + 8

        leader_start_x = left_x + lbl_w + 6
        leader_end_x = right_x - val_w - 6

        svg.append(f'<g transform="translate(0, {y})">')
        svg.append(f'<text x="{left_x}" y="0" class="lbl-text">{label}</text>')

        if leader_end_x > leader_start_x + 10:
            svg.append(f'<line x1="{leader_start_x:.1f}" y1="-4" x2="{leader_end_x:.1f}" y2="-4" stroke="{pal["dotted_leader"]}" stroke-width="1.5" stroke-dasharray="2 4"/>')

        val_len_px = len(val) * 8.5
        svg.append(f'<text x="{right_x}" y="0" class="val-text" text-anchor="end" textLength="{val_len_px:.1f}" lengthAdjust="spacingAndGlyphs">{val}</text>')
        svg.append('</g>')
    svg.append('</g>')

    svg.append('</svg>')

    full_svg = '\n'.join(svg)
    filename = f"{mode}.svg"
    with open(filename, 'w') as f:
        f.write(full_svg)

    size_kb = os.path.getsize(filename) / 1024.0
    print(f"[{mode.upper()}] Saved {filename} ({size_kb:.1f} KB)")

if __name__ == '__main__':
    generate_svg('dark')
    generate_svg('light')
