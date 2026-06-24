data_zs = {
    "our": [74.5, 76.6, 79.1, 79.6, 80.0],
    "splip": [54.8, 69.8, 72.9, 75.6, 76.3],
    "clip_at": [40.8, 51.9, 61.8, 65.9, 72.2],
}

data_gzs = {
    "our": [69.5, 72.2, 76.5, 77.1, 77.6],
    "splip": [41.79, 52.84, 59.86, 63.86, 68.2],
    "clip_at": [21.8, 29.9, 38.8, 48.7, 55.6],
}

classes_zs = {
    "our": [71.2, 76.0, 78.2, 79.1, 80.0],
    "splip": [44.88, 60.81, 69.87, 72.85, 76.3],
    "clip_at": [27.81, 50.87, 61.89, 67.91, 72.2],
}

classes_gzs = {
    "our": [71.0, 74.5, 75.7, 77.1, 77.6],
    "splip": [39.77, 50.76, 58.91, 64.86, 68.2],
    "clip_at": [17.67, 34.78, 46.80, 49.82, 55.6],
}

import matplotlib.pyplot as plt

x = [20, 40, 60, 80, 104]

fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

# ZS-SBIR
axes[0].plot(x, classes_zs["our"], marker="o", label="Our")
axes[0].plot(x, classes_zs["splip"], marker="o", label="SpLIP")
axes[0].plot(x, classes_zs["clip_at"], marker="o", label="CLIP-AT")
axes[0].set_xlabel("ZS-SBIR")
axes[0].set_ylabel("mAP@200")
axes[0].set_xticks(x)
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# GZS-SBIR
axes[1].plot(x, classes_gzs["our"], marker="o", label="Our")
axes[1].plot(x, classes_gzs["splip"], marker="o", label="SpLIP")
axes[1].plot(x, classes_gzs["clip_at"], marker="o", label="CLIP-AT")
axes[1].set_xlabel("GZS-SBIR")
axes[1].set_ylabel("mAP@200")
axes[1].set_xticks(x)
axes[1].legend()
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("classes_zs_gzs.png", dpi=300, bbox_inches="tight")
plt.show()