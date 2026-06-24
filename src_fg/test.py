import torch
from PIL import Image

def generate_perm(num_split = 2):
    return torch.randperm(num_split**2)

def split_img(img, grid=2):
    splitimg = []
    size_img = img.size
    weight = int(size_img[0] // grid)
    height = int(size_img[1] // grid)
    for j in range(grid):
        for k in range(grid):
            box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
            region = img.crop(box)
            splitimg.append(region)
    return splitimg

def rebuild_from_perm(img, perm, grid=2):
    tiles = split_img(img, grid=grid)
    if torch.is_tensor(perm):
        perm = perm.tolist()

    w, h = img.size
    tile_w = w // grid
    tile_h = h // grid

    new_img = Image.new(img.mode, (tile_w * grid, tile_h * grid))

    for i, src_idx in enumerate(perm):
        r = i // grid
        c = i % grid
        new_img.paste(tiles[src_idx], (c * tile_w, r * tile_h))

    return new_img

img = Image.open("n02691156_58-2.png").convert("RGB")
perm = generate_perm(num_split=2)   # torch.randperm(4)
shuffled = rebuild_from_perm(img, perm, grid=2)
shuffled.save("shuffled.png")