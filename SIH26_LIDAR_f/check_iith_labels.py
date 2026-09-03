import os
import numpy as np

PCD = r"C:\Users\Acer\Downloads\IITH_LiDAR_ground_dataset_labelled_raw (2)\IITH_LiDAR_ground_dataset_labelled_raw\labelled_data\slope_0_labelled.pcd"


def decode_rgb(rgb_value):
    rgb = int(float(rgb_value))
    r = (rgb >> 16) & 255
    g = (rgb >> 8) & 255
    b = rgb & 255
    return r, g, b


points = []
colors = []

with open(PCD, "r", encoding="utf-8") as f:
    data = False

    for line in f:
        line = line.strip()

        if line.startswith("DATA"):
            data = True
            continue

        if not data or not line:
            continue

        parts = line.split()

        if len(parts) < 4:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            rgb = decode_rgb(parts[3])

            points.append([x, y, z])
            colors.append(rgb)

        except ValueError:
            pass


points = np.array(points)
colors = np.array(colors)

print("=" * 60)
print("IITH LABEL DIAGNOSTIC")
print("=" * 60)

print("Total points:", len(points))

for colour, name in [((255, 0, 0), "RED"),
                     ((0, 255, 0), "GREEN")]:

    mask = np.all(colors == colour, axis=1)
    xyz = points[mask]

    print()
    print(name)
    print("-" * 40)
    print("Count:", len(xyz))

    if len(xyz) > 0:
        print("X mean:", xyz[:, 0].mean())
        print("Y mean:", xyz[:, 1].mean())
        print("Z min:", xyz[:, 2].min())
        print("Z max:", xyz[:, 2].max())
        print("Z mean:", xyz[:, 2].mean())

print()
print("=" * 60)
print("ALL UNIQUE COLORS")
print("=" * 60)

unique, counts = np.unique(colors, axis=0, return_counts=True)

order = np.argsort(counts)[::-1]

for i in order[:20]:
    print(tuple(unique[i]), "Count:", counts[i])

print("=" * 60)