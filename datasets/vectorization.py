import os
import glob

root = "D:/Research/VLM_project/dataset/TUBerlin"

classes = [
    "helicopter",
        "wrist-watch",
        "dog",
        "mosquito",
        "pear",
        "couch",
        "hammer",
        "car (sedan)",
        "house",
        "baseball bat",
        "toilet",
        "panda",
        "backpack",
        "mug",
        "wineglass",
        "motorbike",
        "eyeglasses",
        "hot air balloon",
        "teapot",
        "shoe",
        "truck",
        "palm tree",
        "cell phone",
        "horse",
        "sailboat",
        "suv",
        "church",
        "floor lamp",
        "bus",
        "tv",
]

total_sketch = 0
total_photo = 0

for cls in classes:
    sketch_dir = os.path.join(root, "sketch", cls)
    photo_dir = os.path.join(root, "photo", cls)

    num_sketch = len(glob.glob(os.path.join(sketch_dir, "*"))) if os.path.exists(sketch_dir) else 0
    num_photo = len(glob.glob(os.path.join(photo_dir, "*"))) if os.path.exists(photo_dir) else 0

    total_sketch += num_sketch
    total_photo += num_photo

    print(f"{cls:20s} | sketch: {num_sketch:4d} | photo: {num_photo:4d}")

print("-" * 50)
print(f"{'TOTAL':20s} | sketch: {total_sketch:4d} | photo: {total_photo:4d}")