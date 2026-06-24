import torch 
def expand_x_by_label(x):
    *tensors, label = x
    max_count = max((label == lb).sum().item() for lb in torch.unique(label))

    expanded_indices = []
    for lb in torch.unique(label):
        idx = (label == lb).nonzero(as_tuple=True)[0]
        n = len(idx)
        expanded_indices.append(
            torch.cat([idx.repeat(max_count // n), idx[:max_count % n]])
        )

    expanded_indices = torch.cat(expanded_indices)
    return tuple(t[expanded_indices] for t in tensors) + (label[expanded_indices],)


label = torch.tensor([2, 1, 3, 2, 1, 2, 3, 1, 2, 2])
photo_tensor = torch.tensor([20, 10, 30, 21, 11, 22, 31, 12, 23, 24])
photo_aug_tensor = torch.tensor([20, 10, 30, 21, 11, 22, 31, 12, 23, 24])
sk_tensor    = torch.tensor([120,110,130,121,111,122,131,112,123,124])

x = photo_tensor, photo_aug_tensor, sk_tensor, label

print("=== Trước ===")
print("photo:", photo_tensor)
print("photo aug:", photo_aug_tensor)
print("sk   :", sk_tensor)
print("label:", label)

x_expand = expand_x_by_label(x)
photo_tensor_e, photo_aug_tensor, sk_tensor_e, label_e = x_expand

print("\n=== Sau ===")
print("photo:", photo_tensor_e)
print("photo aug:", photo_aug_tensor)
print("sk   :", sk_tensor_e)
print("label:", label_e)