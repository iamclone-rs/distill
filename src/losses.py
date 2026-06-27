import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relational_kd_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Relational Knowledge Distillation — dimension-agnostic.
    Match phân phối pairwise cosine similarity (B×B) giữa student và teacher
    qua KL divergence → không yêu cầu D_student == D_teacher.

    student_feat : (B, D_s)  — e.g. 512-dim
    teacher_feat : (B, D_t)  — e.g. 1024-dim (DFN5B)
    """
    s = F.normalize(student_feat.float(), dim=-1)  # (B, D_s)
    t = F.normalize(teacher_feat.float(), dim=-1)  # (B, D_t)
    B = s.shape[0]

    sim_s = (s @ s.T) / temperature   # (B, B)
    sim_t = (t @ t.T) / temperature   # (B, B)

    # Bỏ diagonal (self-similarity không mang thông tin)
    mask = ~torch.eye(B, dtype=torch.bool, device=s.device)
    p_t    = F.softmax(    sim_t[mask].view(B, B - 1), dim=-1)
    log_ps = F.log_softmax(sim_s[mask].view(B, B - 1), dim=-1)

    return F.kl_div(log_ps, p_t, reduction='batchmean')




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


def projected_kd_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    args,
) -> torch.Tensor:
    """
    Strong cross-dim KD after projecting student features into teacher space.
    Combines sample-wise cosine, batch contrastive alignment, and RKD geometry.
    """
    s = F.normalize(student_feat.float(), dim=-1)
    t = F.normalize(teacher_feat.float(), dim=-1)
    temperature = getattr(args, "temperature", 0.07)
    rkd_weight = getattr(args, "rkd_weight", 0.5)
    labels = torch.arange(s.shape[0], device=s.device)

    loss_cos = 1.0 - (s * t).sum(dim=-1).mean()
    logits_st = (s @ t.T) / temperature
    logits_ts = (t @ s.T) / temperature
    loss_contrast = 0.5 * (
        F.cross_entropy(logits_st, labels) +
        F.cross_entropy(logits_ts, labels)
    )
    loss_rkd = relational_kd_loss(student_feat, teacher_feat)

    return loss_cos + loss_contrast + rkd_weight * loss_rkd


def feature_regression_kd(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """
    Sample-wise feature regression KD.
    Use after projection when student and teacher dimensions differ.
    """
    if student_feat.shape[-1] != teacher_feat.shape[-1]:
        raise ValueError(
            "feature regression KD cần student_feat và teacher_feat cùng chiều. "
            "Với teacher 1024-dim, hãy bật --use_distill_proj."
        )

    s = F.normalize(student_feat.float(), dim=-1)
    t = F.normalize(teacher_feat.float(), dim=-1)

    if mode == "cosine":
        return 1.0 - (s * t).sum(dim=-1).mean()
    if mode == "mse":
        return F.mse_loss(s, t)
    if mode == "mae":
        return F.l1_loss(s, t)
    if mode == "smoothl1":
        return F.smooth_l1_loss(s, t)

    raise ValueError(f"Unknown feature regression KD mode: {mode}")


def text_guided_rank_distill(
    student_query: torch.Tensor,
    student_gallery: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_gallery: torch.Tensor,
    student_temperature: float = 0.07,
    teacher_temperature: float = 0.07,
) -> torch.Tensor:
    """
    Distill retrieval ranking, not feature coordinates.

    Student learns the batch-wise sketch->photo ranking distribution.
    Teacher target is built from teacher text(class)->teacher photo similarity,
    so it works even when student is 512-dim and teacher is 1024-dim.
    """
    sq = F.normalize(student_query.float(), dim=-1)
    sg = F.normalize(student_gallery.float(), dim=-1)
    tq = F.normalize(teacher_query.float(), dim=-1)
    tg = F.normalize(teacher_gallery.float(), dim=-1)

    student_logits = (sq @ sg.T) / student_temperature
    with torch.no_grad():
        teacher_logits = (tq @ tg.T) / teacher_temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_probs,
        reduction='batchmean',
    )


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

    # Lựa chọn image distill loss:
    # - auto giữ hành vi cũ.
    # - none tắt image distill để thử các KD khác như rank/text.
    image_distill_mode = getattr(args, "image_distill_mode", "auto")
    zero = torch.tensor(0.0, device=pos_logits.device)
    if image_distill_mode == "none":
        loss_distill_photo = zero
        loss_distill_sk = zero
    elif image_distill_mode == "rkd":
        loss_distill_photo = relational_kd_loss(photo_features, photo_aug_features)
        loss_distill_sk = relational_kd_loss(sk_features, sk_aug_features)
    elif image_distill_mode == "infonce":
        loss_distill_photo = cross_loss(photo_distill_features, photo_aug_features, args)
        loss_distill_sk    = cross_loss(sk_distill_features,    sk_aug_features,    args)
    elif image_distill_mode in ("cosine", "mse", "mae", "smoothl1"):
        loss_distill_photo = feature_regression_kd(
            photo_distill_features,
            photo_aug_features,
            mode=image_distill_mode,
        )
        loss_distill_sk = feature_regression_kd(
            sk_distill_features,
            sk_aug_features,
            mode=image_distill_mode,
        )
    elif image_distill_mode == "auto":
        if getattr(args, "use_distill_proj", False):
            loss_distill_photo = cross_loss(photo_distill_features, photo_aug_features, args)
            loss_distill_sk    = cross_loss(sk_distill_features,    sk_aug_features,    args)
        elif photo_aug_features.shape[-1] != photo_features.shape[-1]:
            loss_distill_photo = relational_kd_loss(photo_features, photo_aug_features)
            loss_distill_sk    = relational_kd_loss(sk_features,    sk_aug_features)
        else:
            loss_distill_photo = cross_loss(photo_features, photo_aug_features, args)
            loss_distill_sk    = cross_loss(sk_features,    sk_aug_features,    args)
    else:
        raise ValueError(f"Unknown image_distill_mode: {image_distill_mode}")
    
    # loss_distill_photo = F.mse_loss(photo_features, photo_aug_features)
    # loss_distill_sk = F.mse_loss(sk_features, sk_aug_features)
    
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
    # photo_score = cos(photo_features, photo_aug_features)
    # sketch_score = cos(sk_features, sk_aug_features)
    # loss_distill_photo = 1.0 - torch.mean(photo_score)
    # loss_distill_sk = 1.0 - torch.mean(sketch_score)
    
    # loss_distill_photo = F.l1_loss(photo_features, photo_aug_features)
    # loss_distill_sk = F.l1_loss(sk_features, sk_aug_features)
    
    if getattr(args, "distill_photo_only", False):
        lambda_sketch_distill = getattr(args, "lambda_sketch_distill", 0.0)
        loss_distill = loss_distill_photo + lambda_sketch_distill * loss_distill_sk
    else:
        loss_distill = loss_distill_sk + loss_distill_photo 

    loss_text_distill = torch.tensor(0.0, device=pos_logits.device)
    if (
        getattr(args, "distill_text", False)
        and teacher_text_features is not None
        and photo_text_distill_features is not None
        and sk_text_distill_features is not None
    ):
        text_distill_mode = getattr(args, "text_distill_mode", "infonce")
        if text_distill_mode == "infonce":
            loss_text_distill = (
                cross_loss(photo_text_distill_features, teacher_text_features, args)
                + cross_loss(sk_text_distill_features, teacher_text_features, args)
            )
        elif text_distill_mode in ("cosine", "mse", "mae", "smoothl1"):
            loss_text_distill = (
                feature_regression_kd(
                    photo_text_distill_features,
                    teacher_text_features,
                    mode=text_distill_mode,
                )
                + feature_regression_kd(
                    sk_text_distill_features,
                    teacher_text_features,
                    mode=text_distill_mode,
                )
            )
        else:
            raise ValueError(f"Unknown text_distill_mode: {text_distill_mode}")

    loss_rank_distill = torch.tensor(0.0, device=pos_logits.device)
    if getattr(args, "distill_rank", False):
        if teacher_text_features is None:
            raise ValueError("distill_rank cần strong teacher có text encoder, ví dụ --teacher dfn5b hoặc --teacher laion_h.")
        teacher_query = teacher_text_features[label]
        loss_rank_distill = text_guided_rank_distill(
            student_query=sk_features,
            student_gallery=photo_features,
            teacher_query=teacher_query,
            teacher_gallery=photo_aug_features,
            student_temperature=getattr(args, "rank_distill_temperature", 0.07),
            teacher_temperature=getattr(args, "teacher_rank_temperature", 0.07),
        )
    
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, margin=0.2)
    if getattr(args, "infer_with_distill_proj", False):
        photo_triplet_features = F.normalize(photo_distill_features.float(), dim=-1)
        sk_triplet_features = F.normalize(sk_distill_features.float(), dim=-1)
        neg_triplet_features = F.normalize(neg_distill_features.float(), dim=-1)
        loss_triplet = triplet(sk_triplet_features, photo_triplet_features, neg_triplet_features)
    else:
        loss_triplet = triplet(sk_feature_norm, photo_features_norm, neg_features)
    
    if getattr(args, "infer_with_distill_proj", False):
        nt_xent_loss = nt_xent(photo_distill_features, sk_distill_features)
    else:
        nt_xent_loss = nt_xent(photo_features, sk_features)
    
    # lambda_distill: điều chỉnh trọng số loss distillation
    # - clip32 teacher: lambda_distill=1.0 là hợp lý (scale tương đương cls/nt_xent)
    # - dfn5b  teacher: lambda_distill cần lớn hơn (~10–50) vì RKD scale nhỏ hơn
    #   gợi ý: bắt đầu với lambda_distill=10.0 khi dùng dfn5b
    lambda_distill = getattr(args, 'lambda_distill', 1.0)
    lambda_text_distill = getattr(args, 'lambda_text_distill', 1.0)
    lambda_rank_distill = getattr(args, 'lambda_rank_distill', 1.0)
    
    total_loss = (
        loss_cls \
        + loss_triplet \
        + lambda_distill * loss_distill \
        + lambda_text_distill * loss_text_distill \
        + lambda_rank_distill * loss_rank_distill \
        + nt_xent_loss
    )
    
    return total_loss
