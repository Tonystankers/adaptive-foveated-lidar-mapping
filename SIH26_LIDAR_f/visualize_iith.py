import os
import numpy as np
import matplotlib.pyplot as plt

PCD = r"C:\Users\Acer\Downloads\IITH_LiDAR_ground_dataset_labelled_raw (2)\IITH_LiDAR_ground_dataset_labelled_raw\labelled_data\slope_0_labelled.pcd"


def decode_rgb(value):
    rgb = int(float(value))
    r = (rgb >> 16) & 255
    g = (rgb >> 8) & 255
    b = rgb & 255
    return r, g, b


points = []
colors = []

with open(PCD, "r", encoding="utf-8") as f:

    data_started = False

    for line in f:

        line = line.strip()

        if not line:
            continue

        if line.startswith("DATA"):
            data_started = True
            continue

        if not data_started:
            continue

        parts = line.split()

        if len(parts) < 4:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            rgb = float(parts[3])

            r, g, b = decode_rgb(rgb)

            points.append([x, y, z])
            colors.append([r / 255, g / 255, b / 255])

        except ValueError:
            continue


points = np.asarray(points)
colors = np.asarray(colors)

print("Points:", len(points))
print("Opening visualization...")

plt.figure(figsize=(10, 8))

plt.scatter(
    points[:, 0],
    points[:, 1],
    c=colors,
    s=2
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("IITH LiDAR - slope_0")
plt.axis("equal")

plt.show()