import os

root_dir = "D:/Research/VLM_project/dataset/TUBerlin/photo"  # ví dụ: "/kaggle/input/datasets/.../photo"
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)

    if os.path.isdir(folder_path):
        num_images = sum(
            1 for file_name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, file_name))
            and os.path.splitext(file_name)[1].lower() in valid_exts
        )

        if num_images < 400:
            print(f"{folder_name:<20} | photo: {num_images:4d}")