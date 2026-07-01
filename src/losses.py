import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kd_div_loss(student_feat1, student_feat2, teacher_feat1, teacher_feat2, temperature=0.07):
    """
    KL-divergence distillation trên cosine similarity matrix.
    Ép phân bố quan hệ giữa hai tập feature của student giống teacher.
    """
    s1 = F.normalize(student_feat1, dim=-1)
    s2 = F.normalize(student_feat2, dim=-1)
    t1 = F.normalize(teacher_feat1, dim=-1)
    t2 = F.normalize(teacher_feat2, dim=-1)

    sim_s = (s1 @ s2.t()) / temperature
    log_p_s = F.log_softmax(sim_s, dim=-1)

    with torch.no_grad():
        sim_t = (t1 @ t2.t()) / temperature
        p_t = F.softmax(sim_t, dim=-1)

    return F.kl_div(log_p_s, p_t, reduction='batchmean')


def add_kd_div(loss_distill, loss_dict, name, weight, student_feat1, student_feat2, teacher_feat1, teacher_feat2, temperature):
    if weight <= 0 or student_feat1 is None or student_feat2 is None or teacher_feat1 is None or teacher_feat2 is None:
        return loss_distill

    loss_value = kd_div_loss(
        student_feat1,
        student_feat2,
        teacher_feat1,
        teacher_feat2,
        temperature,
    )
    loss_distill = loss_distill + weight * loss_value
    loss_dict[name] = loss_value
    return loss_distill


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
        neg_features, label, pos_logits, sk_logits, photo_features, sk_features = features[:10]

    if len(features) >= 13:
        photo_distill_features, sk_distill_features, neg_distill_features = features[10:13]
    else:
        photo_distill_features = photo_features
        sk_distill_features = sk_features
        neg_distill_features = neg_features

    if len(features) >= 16:
        photo_text_distill_features, sk_text_distill_features, teacher_text_features = features[13:16]
    else:
        photo_text_distill_features = None
        sk_text_distill_features = None
        teacher_text_features = None

    label = label.to(pos_logits.device)
    loss_ce_photo = F.cross_entropy(pos_logits, label)
    loss_ce_sk = F.cross_entropy(sk_logits, label)
    loss_cls = loss_ce_photo + loss_ce_sk
    loss_dict = {}
    loss_distill = torch.tensor(0.0, device=pos_logits.device)

    temp = getattr(args, "rkd_temperature", 0.07)
    loss_distill = add_kd_div(
        loss_distill,
        loss_dict,
        "kd_sk_ph",
        getattr(args, "lambda_rkd_sk_ph", 0.0),
        sk_distill_features,
        photo_distill_features,
        sk_aug_features,
        photo_aug_features,
        temp,
    )
    loss_distill = add_kd_div(
        loss_distill,
        loss_dict,
        "kd_ph_txt",
        getattr(args, "lambda_rkd_ph_txt", 0.0),
        photo_distill_features,
        photo_text_distill_features,
        photo_aug_features,
        teacher_text_features,
        temp,
    )
    loss_distill = add_kd_div(
        loss_distill,
        loss_dict,
        "kd_sk_txt",
        getattr(args, "lambda_rkd_sk_txt", 0.0),
        sk_distill_features,
        sk_text_distill_features,
        sk_aug_features,
        teacher_text_features,
        temp,
    )

    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, margin=0.2)
    loss_triplet = triplet(sk_feature_norm, photo_features_norm, neg_features)
    
    nt_xent_loss = nt_xent(photo_features, sk_features)

    lambda_cls = getattr(args, "lambda_cls", 1.0)
    lambda_triplet = getattr(args, "lambda_triplet", 1.0)
    lambda_nt_xent = getattr(args, "lambda_nt_xent", 1.0)
    
    total_loss = (
        lambda_cls * loss_cls \
        + lambda_triplet * loss_triplet \
        + loss_distill \
        + lambda_nt_xent * nt_xent_loss
    )
    
    loss_dict['cls'] = loss_cls
    loss_dict['triplet'] = loss_triplet
    loss_dict['nt_xent'] = nt_xent_loss
    
    return total_loss, loss_dict
