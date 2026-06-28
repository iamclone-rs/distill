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


def soft_infonce_kd(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    student_temperature: float = 0.07,
    teacher_temperature: float = 0.07,
    teacher_weight: float = 0.5,
) -> torch.Tensor:
    """
    Teacher-weighted InfoNCE.

    Hard InfoNCE says only the paired item is positive. This loss blends that
    hard target with a teacher-teacher similarity distribution over the batch.
    """
    if student_feat.shape[-1] != teacher_feat.shape[-1]:
        raise ValueError(
            "soft_infonce cần student_feat và teacher_feat cùng chiều. "
            "Với teacher 1024-dim, hãy bật --use_distill_proj hoặc --distill_text."
        )

    s = F.normalize(student_feat.float(), dim=-1)
    t = F.normalize(teacher_feat.float(), dim=-1)
    batch_size = s.shape[0]
    labels = torch.arange(batch_size, device=s.device)

    hard_target = F.one_hot(labels, num_classes=batch_size).float()
    teacher_weight = max(0.0, min(1.0, float(teacher_weight)))

    with torch.no_grad():
        teacher_logits = (t @ t.T) / teacher_temperature
        teacher_target = F.softmax(teacher_logits, dim=-1)
        target = teacher_weight * teacher_target + (1.0 - teacher_weight) * hard_target

    logits_st = (s @ t.T) / student_temperature
    logits_ts = (t @ s.T) / student_temperature
    loss_st = F.kl_div(F.log_softmax(logits_st, dim=-1), target, reduction='batchmean')
    loss_ts = F.kl_div(F.log_softmax(logits_ts, dim=-1), target.T, reduction='batchmean')

    return 0.5 * (loss_st + loss_ts)


def pair_infonce_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    One-way InfoNCE: student queries choose the paired teacher feature.
    """
    if student_feat.shape[-1] != teacher_feat.shape[-1]:
        raise ValueError(
            "pair_infonce cần student_feat và teacher_feat cùng chiều. "
            "Với teacher 1024-dim, hãy bật --use_distill_proj hoặc --distill_text."
        )

    s = F.normalize(student_feat.float(), dim=-1)
    t = F.normalize(teacher_feat.float(), dim=-1)
    labels = torch.arange(s.shape[0], device=s.device)
    logits = (s @ t.T) / temperature
    return F.cross_entropy(logits, labels)


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


def cross_modal_matrix_distill(
    student_query: torch.Tensor,
    student_gallery: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_gallery: torch.Tensor,
    mode: str = "smoothl1",
) -> torch.Tensor:
    """
    Match the actual sketch->photo similarity matrix.

    Student matrix: sketch_student @ photo_student.T
    Teacher matrix: teacher_text[label] @ teacher_photo.T
    """
    sq = F.normalize(student_query.float(), dim=-1)
    sg = F.normalize(student_gallery.float(), dim=-1)
    tq = F.normalize(teacher_query.float(), dim=-1)
    tg = F.normalize(teacher_gallery.float(), dim=-1)

    student_sim = sq @ sg.T
    with torch.no_grad():
        teacher_sim = tq @ tg.T

    if mode == "mse":
        return F.mse_loss(student_sim, teacher_sim)
    if mode == "smoothl1":
        return F.smooth_l1_loss(student_sim, teacher_sim)

    raise ValueError(f"Unknown xmodal_distill_mode: {mode}")


def class_aware_listwise_distill(
    student_query: torch.Tensor,
    student_gallery: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_gallery: torch.Tensor,
    labels: torch.Tensor,
    student_temperature: float = 0.07,
    teacher_temperature: float = 0.07,
    teacher_weight: float = 0.5,
    bidirectional: bool = False,
) -> torch.Tensor:
    """
    Listwise sketch->photo distillation for category-level SBIR.

    It fixes diagonal InfoNCE's weak point: all gallery photos with the same
    class label are positives, not only the item at the same batch index.
    """
    sq = F.normalize(student_query.float(), dim=-1)
    sg = F.normalize(student_gallery.float(), dim=-1)
    tq = F.normalize(teacher_query.float(), dim=-1)
    tg = F.normalize(teacher_gallery.float(), dim=-1)

    labels = labels.to(student_query.device)
    pos_mask = labels[:, None].eq(labels[None, :]).float()
    class_target = pos_mask / pos_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    teacher_weight = max(0.0, min(1.0, float(teacher_weight)))
    student_logits = (sq @ sg.T) / student_temperature
    with torch.no_grad():
        teacher_logits = (tq @ tg.T) / teacher_temperature
        teacher_target = F.softmax(teacher_logits, dim=-1)
        target = teacher_weight * teacher_target + (1.0 - teacher_weight) * class_target

    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        target,
        reduction='batchmean',
    )

    if bidirectional:
        class_target_t = pos_mask.T / pos_mask.T.sum(dim=-1, keepdim=True).clamp_min(1.0)
        student_logits_t = (sg @ sq.T) / student_temperature
        with torch.no_grad():
            teacher_logits_t = (tg @ tq.T) / teacher_temperature
            teacher_target_t = F.softmax(teacher_logits_t, dim=-1)
            target_t = teacher_weight * teacher_target_t + (1.0 - teacher_weight) * class_target_t
        loss = 0.5 * (
            loss
            + F.kl_div(
                F.log_softmax(student_logits_t, dim=-1),
                target_t,
                reduction='batchmean',
            )
        )

    return loss


def cross_modal_supcon_loss(
    query_feat: torch.Tensor,
    gallery_feat: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    bidirectional: bool = False,
) -> torch.Tensor:
    """
    Cross-modal supervised contrastive loss for category-level SBIR.

    For each sketch query, every photo with the same class label is positive.
    """
    q = F.normalize(query_feat.float(), dim=-1)
    g = F.normalize(gallery_feat.float(), dim=-1)
    labels = labels.to(query_feat.device)

    pos_mask = labels[:, None].eq(labels[None, :]).float()
    logits = (q @ g.T) / temperature
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -((pos_mask * log_prob).sum(dim=-1) / pos_mask.sum(dim=-1).clamp_min(1.0)).mean()

    if bidirectional:
        logits_t = (g @ q.T) / temperature
        log_prob_t = F.log_softmax(logits_t, dim=-1)
        pos_mask_t = pos_mask.T
        loss_t = -((pos_mask_t * log_prob_t).sum(dim=-1) / pos_mask_t.sum(dim=-1).clamp_min(1.0)).mean()
        loss = 0.5 * (loss + loss_t)

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
    elif image_distill_mode == "pair_infonce":
        loss_distill_photo = pair_infonce_loss(
            photo_distill_features,
            photo_aug_features,
            temperature=getattr(args, "temperature", 0.07),
        )
        loss_distill_sk = pair_infonce_loss(
            sk_distill_features,
            sk_aug_features,
            temperature=getattr(args, "temperature", 0.07),
        )
    elif image_distill_mode == "soft_infonce":
        loss_distill_photo = soft_infonce_kd(
            photo_distill_features,
            photo_aug_features,
            student_temperature=getattr(args, "soft_infonce_temperature", 0.07),
            teacher_temperature=getattr(args, "teacher_soft_infonce_temperature", 0.07),
            teacher_weight=getattr(args, "soft_infonce_teacher_weight", 0.5),
        )
        loss_distill_sk = soft_infonce_kd(
            sk_distill_features,
            sk_aug_features,
            student_temperature=getattr(args, "soft_infonce_temperature", 0.07),
            teacher_temperature=getattr(args, "teacher_soft_infonce_temperature", 0.07),
            teacher_weight=getattr(args, "soft_infonce_teacher_weight", 0.5),
        )
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
    
    lambda_photo_distill = getattr(args, "lambda_photo_distill", None)
    lambda_sketch_distill = getattr(args, "lambda_sketch_distill", 0.0)
    lambda_distill = getattr(args, 'lambda_distill', 1.0)

    if lambda_photo_distill is not None:
        loss_distill = (
            lambda_photo_distill * loss_distill_photo
            + lambda_sketch_distill * loss_distill_sk
        )
    elif getattr(args, "distill_photo_only", False):
        loss_distill = lambda_distill * (
            loss_distill_photo + lambda_sketch_distill * loss_distill_sk
        )
    else:
        loss_distill = lambda_distill * (loss_distill_sk + loss_distill_photo)

    lambda_photo_rkd = getattr(args, "lambda_photo_rkd", 0.0)
    lambda_sketch_rkd = getattr(args, "lambda_sketch_rkd", 0.0)
    loss_aux_image_rkd = torch.tensor(0.0, device=pos_logits.device)
    if lambda_photo_rkd > 0:
        loss_aux_image_rkd = loss_aux_image_rkd + lambda_photo_rkd * relational_kd_loss(
            photo_features,
            photo_aug_features,
        )
    if lambda_sketch_rkd > 0:
        loss_aux_image_rkd = loss_aux_image_rkd + lambda_sketch_rkd * relational_kd_loss(
            sk_features,
            sk_aug_features,
        )

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
        elif text_distill_mode == "pair_infonce":
            loss_text_distill = (
                pair_infonce_loss(
                    photo_text_distill_features,
                    teacher_text_features,
                    temperature=getattr(args, "temperature", 0.07),
                )
                + pair_infonce_loss(
                    sk_text_distill_features,
                    teacher_text_features,
                    temperature=getattr(args, "temperature", 0.07),
                )
            )
        elif text_distill_mode == "soft_infonce":
            loss_text_distill = (
                soft_infonce_kd(
                    photo_text_distill_features,
                    teacher_text_features,
                    student_temperature=getattr(args, "soft_infonce_temperature", 0.07),
                    teacher_temperature=getattr(args, "teacher_soft_infonce_temperature", 0.07),
                    teacher_weight=getattr(args, "soft_infonce_teacher_weight", 0.5),
                )
                + soft_infonce_kd(
                    sk_text_distill_features,
                    teacher_text_features,
                    student_temperature=getattr(args, "soft_infonce_temperature", 0.07),
                    teacher_temperature=getattr(args, "teacher_soft_infonce_temperature", 0.07),
                    teacher_weight=getattr(args, "soft_infonce_teacher_weight", 0.5),
                )
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
        elif text_distill_mode == "rkd":
            loss_text_distill = (
                relational_kd_loss(photo_text_distill_features, teacher_text_features)
                + relational_kd_loss(sk_text_distill_features, teacher_text_features)
            )
        else:
            raise ValueError(f"Unknown text_distill_mode: {text_distill_mode}")

    lambda_text_rkd = getattr(args, "lambda_text_rkd", 0.0)
    loss_aux_text_rkd = torch.tensor(0.0, device=pos_logits.device)
    if lambda_text_rkd > 0:
        if teacher_text_features is None:
            raise ValueError("lambda_text_rkd cần strong teacher có text encoder, ví dụ --teacher dfn5b hoặc --teacher laion_h.")
        if photo_text_distill_features is None or sk_text_distill_features is None:
            raise ValueError("lambda_text_rkd cần text features từ model forward.")
        loss_aux_text_rkd = lambda_text_rkd * (
            relational_kd_loss(photo_text_distill_features, teacher_text_features)
            + relational_kd_loss(sk_text_distill_features, teacher_text_features)
        )

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

    loss_xmodal_distill = torch.tensor(0.0, device=pos_logits.device)
    if getattr(args, "distill_xmodal", False):
        if teacher_text_features is None:
            raise ValueError("distill_xmodal cần strong teacher có text encoder, ví dụ --teacher dfn5b hoặc --teacher laion_h.")
        teacher_query = teacher_text_features[label]
        loss_xmodal_distill = cross_modal_matrix_distill(
            student_query=sk_features,
            student_gallery=photo_features,
            teacher_query=teacher_query,
            teacher_gallery=photo_aug_features,
            mode=getattr(args, "xmodal_distill_mode", "smoothl1"),
        )

    loss_listwise_distill = torch.tensor(0.0, device=pos_logits.device)
    if getattr(args, "distill_listwise", False):
        if teacher_text_features is None:
            raise ValueError("distill_listwise cần strong teacher có text encoder, ví dụ --teacher dfn5b hoặc --teacher laion_h.")
        teacher_query = teacher_text_features[label]
        loss_listwise_distill = class_aware_listwise_distill(
            student_query=sk_features,
            student_gallery=photo_features,
            teacher_query=teacher_query,
            teacher_gallery=photo_aug_features,
            labels=label,
            student_temperature=getattr(args, "listwise_distill_temperature", 0.07),
            teacher_temperature=getattr(args, "teacher_listwise_temperature", 0.07),
            teacher_weight=getattr(args, "listwise_teacher_weight", 0.5),
            bidirectional=getattr(args, "listwise_bidirectional", False),
        )

    loss_xsupcon = torch.tensor(0.0, device=pos_logits.device)
    if getattr(args, "distill_xsupcon", False):
        loss_xsupcon = cross_modal_supcon_loss(
            query_feat=sk_features,
            gallery_feat=photo_features,
            labels=label,
            temperature=getattr(args, "xsupcon_temperature", 0.07),
            bidirectional=getattr(args, "xsupcon_bidirectional", False),
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
    
    lambda_text_distill = getattr(args, 'lambda_text_distill', 1.0)
    lambda_rank_distill = getattr(args, 'lambda_rank_distill', 1.0)
    lambda_xmodal_distill = getattr(args, 'lambda_xmodal_distill', 1.0)
    lambda_listwise_distill = getattr(args, 'lambda_listwise_distill', 1.0)
    lambda_xsupcon = getattr(args, 'lambda_xsupcon', 1.0)
    
    total_loss = (
        loss_cls \
        + loss_triplet \
        + loss_distill \
        + loss_aux_image_rkd \
        + lambda_text_distill * loss_text_distill \
        + loss_aux_text_rkd \
        + lambda_rank_distill * loss_rank_distill \
        + lambda_xmodal_distill * loss_xmodal_distill \
        + lambda_listwise_distill * loss_listwise_distill \
        + lambda_xsupcon * loss_xsupcon \
        + nt_xent_loss
    )
    
    return total_loss
