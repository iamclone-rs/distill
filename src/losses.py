import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relational_kd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """
    Relational Knowledge Distillation (RKD) — dimension agnostic.

    Thay vì ép student match trực tiếp teacher features (yêu cầu cùng dim),
    RKD học cấu trúc tương quan pairwise giữa các sample trong batch.

    Cụ thể: match phân phối softmax của ma trận cosine similarity (B×B)
    giữa student và teacher qua KL divergence.

    Args:
        student_feat : (B, D_s) — features của student (vd: 512-dim)
        teacher_feat : (B, D_t) — features của teacher (vd: 1024-dim), không cần D_s == D_t
        temperature  : scale similarity trước softmax (giá trị lớn → softer distribution)

    Returns:
        Scalar loss.
    """
    # Normalize về unit sphere trước khi tính cosine similarity
    s = F.normalize(student_feat.float(), dim=-1)  # (B, D_s)
    t = F.normalize(teacher_feat.float(), dim=-1)  # (B, D_t)

    B = s.shape[0]

    # Pairwise cosine similarity matrices — cùng shape (B, B) dù D_s ≠ D_t
    sim_s = (s @ s.T) / temperature  # (B, B)
    sim_t = (t @ t.T) / temperature  # (B, B)

    # Loại bỏ diagonal (self-similarity = 1, không mang thông tin)
    mask = ~torch.eye(B, dtype=torch.bool, device=s.device)
    sim_s_off = sim_s[mask].view(B, B - 1)  # (B, B-1)
    sim_t_off = sim_t[mask].view(B, B - 1)  # (B, B-1)

    # Teacher distribution: softmax trên hàng — học "ai gần ai"
    p_t = F.softmax(sim_t_off, dim=-1)           # (B, B-1)
    # Student log-distribution
    log_p_s = F.log_softmax(sim_s_off, dim=-1)   # (B, B-1)

    # KL(teacher || student) — student học để match teacher's relational structure
    return F.kl_div(log_p_s, p_t, reduction='batchmean')

def cross_loss(feature_1, feature_2, args):
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature

    return nn.CrossEntropyLoss()(logits, labels)


def nt_xent(features_view1: torch.Tensor, features_view2: torch.Tensor):
    """
    NT-Xent for SimCLR
    features_view1, features_view2: (B, D)
    """
    features_view1 = F.normalize(features_view1)
    features_view2 = F.normalize(features_view2)
    
    temperature = 0.07
    B, D = features_view1.shape
    device = features_view1.device

    z = torch.cat([features_view1, features_view2], dim=0)

    logits = z @ z.t()                              # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float('-inf'))

    logits = logits / temperature

    labels = torch.cat([
        torch.arange(B, 2*B, device=device),
        torch.arange(0, B, device=device)
    ], dim=0).long()

    loss = F.cross_entropy(logits, labels)
    return loss


def loss_fn(args, model, features, mode='train'):
    photo_features_norm, sk_feature_norm, photo_aug_features, sk_aug_features, \
        neg_features, label, pos_logits, sk_logits, photo_features, sk_features = features

    label = label.to(pos_logits.device)
    loss_ce_photo = F.cross_entropy(pos_logits, label)
    loss_ce_sk = F.cross_entropy(sk_logits, label)
    loss_cls = loss_ce_photo + loss_ce_sk
    
    # Nếu teacher (DFN5B 1024-dim) và student (512-dim) khác dimension
    # → dùng Relational KD (dimension-agnostic) thay cross_loss
    if photo_aug_features.shape[-1] != photo_features.shape[-1]:
        loss_distill_photo = relational_kd_loss(photo_features, photo_aug_features)
        loss_distill_sk    = relational_kd_loss(sk_features,    sk_aug_features)
    else:
        loss_distill_photo = cross_loss(photo_features, photo_aug_features, args)
        loss_distill_sk    = cross_loss(sk_features,    sk_aug_features, args)
    
    # loss_distill_photo = F.mse_loss(photo_features, photo_aug_features)
    # loss_distill_sk = F.mse_loss(sk_features, sk_aug_features)
    
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
    # photo_score = cos(photo_features, photo_aug_features)
    # sketch_score = cos(sk_features, sk_aug_features)
    # loss_distill_photo = 1.0 - torch.mean(photo_score)
    # loss_distill_sk = 1.0 - torch.mean(sketch_score)
    
    # loss_distill_photo = F.l1_loss(photo_features, photo_aug_features)
    # loss_distill_sk = F.l1_loss(sk_features, sk_aug_features)
    
    loss_distill = loss_distill_sk + loss_distill_photo 
    
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, margin=0.2)
    loss_triplet = triplet(sk_feature_norm, photo_features_norm, neg_features)
    
    nt_xent_loss = nt_xent(photo_features, sk_features)
    
    total_loss = (
        loss_cls \
        + loss_triplet \
        + loss_distill \
        + nt_xent_loss
    )
    
    return total_loss