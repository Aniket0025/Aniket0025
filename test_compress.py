import numpy as np

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

# Test with 15000 dots
dots = np.random.randint(0, 300, size=(15000, 2))
path_str = dots_to_stroke_runs(dots, 70, 142, 1.06)
print("Compressed path length for 15000 dots:", len(path_str), "bytes =", round(len(path_str)/1024, 1), "KB")
