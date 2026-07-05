import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kd_div_loss(student_feat1, student_feat2, teacher_feat1, teacher_feat2, temperature=0.07):
    """
    KL-divergence distillation trên cosine similarity matrix.
    Ép phân bố quan hệ giữa hai tập feature của student giống teacher.
    """
    # Similarity distributions are more stable in FP32. This also handles mixed
    # teacher branches, e.g. an FP32 image adapter with an FP16 text encoder.
    student_device = student_feat1.device
    s1 = F.normalize(
        student_feat1.to(device=student_device, dtype=torch.float32), dim=-1
    )
    s2 = F.normalize(
        student_feat2.to(device=student_device, dtype=torch.float32), dim=-1
    )
    t1 = F.normalize(
        teacher_feat1.to(device=student_device, dtype=torch.float32), dim=-1
    )
    t2 = F.normalize(
        teacher_feat2.to(device=student_device, dtype=torch.float32), dim=-1
    )

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


def adapter_guided_ranking_loss(
    student_sketch,
    student_photo,
    teacher_sketch_adapted,
    teacher_photo_adapted,
    teacher_sketch_base,
    teacher_photo_base,
    labels,
    temperature=0.07,
    margin=0.2,
):
    """Rank positives above adapter-improved, teacher-hard negatives."""
    device = student_sketch.device
    student_sketch = F.normalize(student_sketch.float(), dim=-1)
    student_photo = F.normalize(
        student_photo.to(device=device, dtype=torch.float32), dim=-1
    )
    labels = labels.to(device)

    sim_student = student_sketch @ student_photo.t()
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = ~positive_mask
    pair_mask = positive_mask[:, :, None] & negative_mask[:, None, :]

    with torch.no_grad():
        teacher_sketch_adapted = F.normalize(
            teacher_sketch_adapted.to(device=device, dtype=torch.float32), dim=-1
        )
        teacher_photo_adapted = F.normalize(
            teacher_photo_adapted.to(device=device, dtype=torch.float32), dim=-1
        )
        teacher_sketch_base = F.normalize(
            teacher_sketch_base.to(device=device, dtype=torch.float32), dim=-1
        )
        teacher_photo_base = F.normalize(
            teacher_photo_base.to(device=device, dtype=torch.float32), dim=-1
        )

        sim_adapted = teacher_sketch_adapted @ teacher_photo_adapted.t()
        sim_base = teacher_sketch_base @ teacher_photo_base.t()

        margin_adapted = sim_adapted[:, :, None] - sim_adapted[:, None, :]
        margin_base = sim_base[:, :, None] - sim_base[:, None, :]
        adapter_gain = (margin_adapted - margin_base).clamp_min(0.0)

        negative_logits = (sim_adapted / temperature).masked_fill(
            ~negative_mask, -1e9
        )
        hard_negative_weight = F.softmax(negative_logits, dim=-1)
        hard_negative_weight = hard_negative_weight * negative_mask.float()
        hard_negative_weight = hard_negative_weight / hard_negative_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        pair_weight = (
            adapter_gain
            * hard_negative_weight[:, None, :]
            * pair_mask.float()
        )
        weight_sum = pair_weight.sum(dim=(1, 2))
        valid_query = weight_sum > 1e-12
        pair_weight = pair_weight / weight_sum[:, None, None].clamp_min(1e-12)

    student_pair_loss = F.softplus(
        margin + sim_student[:, None, :] - sim_student[:, :, None]
    )
    query_loss = (pair_weight * student_pair_loss).sum(dim=(1, 2))
    if valid_query.any():
        return query_loss[valid_query].mean()
    return sim_student.sum() * 0.0


def infonce_distill_loss(student_feat, teacher_feat, temperature=0.07):
    teacher_feat = teacher_feat.to(dtype=student_feat.dtype, device=student_feat.device)
    student_feat = F.normalize(student_feat, dim=1)
    teacher_feat = F.normalize(teacher_feat, dim=1)
    features = torch.cat((student_feat, teacher_feat), dim=0)

    labels = torch.cat([torch.arange(len(student_feat), device=student_feat.device) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    similarity_matrix = features @ features.t()
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=student_feat.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=student_feat.device)
    return F.cross_entropy(logits, targets)


def independent_cosine_distill_loss(student_feat, teacher_feat):
    """Cosine feature alignment theo từng phần tử, không dùng negatives hay quan hệ toàn batch."""
    teacher_feat = teacher_feat.to(dtype=student_feat.dtype, device=student_feat.device)
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    return (1.0 - (student_feat * teacher_feat).sum(dim=-1)).mean()


def add_independent_cosine(loss_distill, loss_dict, name, weight, student_feat, teacher_feat):
    if weight <= 0 or student_feat is None or teacher_feat is None:
        return loss_distill

    loss_value = independent_cosine_distill_loss(student_feat, teacher_feat)
    loss_distill = loss_distill + weight * loss_value
    loss_dict[name] = loss_value
    return loss_distill


def teacher_weighted_ntxent_loss(
    student_sketch_feat,
    student_photo_feat,
    teacher_sketch_feat,
    teacher_photo_feat,
    alpha=0.3,
    temperature=0.08,
):
    """
    NT-Xent sketch-photo với target mềm từ teacher.

    alpha=0   -> chỉ dùng target cứng theo diagonal như contrastive CE.
    alpha=1   -> hoàn toàn học phân phối similarity của teacher.
    0<alpha<1 -> vừa giữ positive đúng sample, vừa cho teacher định hình hard negatives.
    """
    if (
        student_sketch_feat is None
        or student_photo_feat is None
        or teacher_sketch_feat is None
        or teacher_photo_feat is None
    ):
        return None

    alpha = max(0.0, min(1.0, float(alpha)))
    student_sketch_feat = F.normalize(student_sketch_feat, dim=-1)
    student_photo_feat = F.normalize(student_photo_feat, dim=-1)
    teacher_sketch_feat = F.normalize(
        teacher_sketch_feat.to(device=student_sketch_feat.device, dtype=student_sketch_feat.dtype),
        dim=-1,
    )
    teacher_photo_feat = F.normalize(
        teacher_photo_feat.to(device=student_photo_feat.device, dtype=student_photo_feat.dtype),
        dim=-1,
    )

    logits_sk_ph = student_sketch_feat @ student_photo_feat.t() / temperature
    logits_ph_sk = logits_sk_ph.t()

    with torch.no_grad():
        teacher_logits = teacher_sketch_feat @ teacher_photo_feat.t() / temperature
        teacher_target_sk_ph = F.softmax(teacher_logits, dim=-1)
        teacher_target_ph_sk = F.softmax(teacher_logits.t(), dim=-1)

        batch_size = student_sketch_feat.shape[0]
        hard_target = torch.eye(batch_size, device=student_sketch_feat.device, dtype=student_sketch_feat.dtype)
        target_sk_ph = (1.0 - alpha) * hard_target + alpha * teacher_target_sk_ph
        target_ph_sk = (1.0 - alpha) * hard_target + alpha * teacher_target_ph_sk

    loss_sk_ph = -(target_sk_ph * F.log_softmax(logits_sk_ph, dim=-1)).sum(dim=-1).mean()
    loss_ph_sk = -(target_ph_sk * F.log_softmax(logits_ph_sk, dim=-1)).sum(dim=-1).mean()
    return 0.5 * (loss_sk_ph + loss_ph_sk)


def add_infonce_distill(loss_distill, loss_dict, name, weight, student_feat, teacher_feat, temperature):
    if weight <= 0 or student_feat is None or teacher_feat is None:
        return loss_distill

    loss_value = infonce_distill_loss(student_feat, teacher_feat, temperature)
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

    if len(features) >= 18:
        photo_teacher_base, sketch_teacher_base = features[16:18]
    else:
        photo_teacher_base = None
        sketch_teacher_base = None

    label = label.to(pos_logits.device)
    loss_ce_photo = F.cross_entropy(pos_logits, label)
    loss_ce_sk = F.cross_entropy(sk_logits, label)
    loss_cls = loss_ce_photo + loss_ce_sk
    loss_dict = {}
    loss_distill = torch.tensor(0.0, device=pos_logits.device)

    distill_mode = getattr(args, "distill_mode", "kd_div")
    if distill_mode == "kd_div":
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
    elif distill_mode == "linear_infonce":
        temp = getattr(args, "infonce_temperature", getattr(args, "temperature", 0.07))
        loss_distill = add_infonce_distill(
            loss_distill,
            loss_dict,
            "infonce_photo",
            getattr(args, "lambda_infonce_photo", 0.0),
            photo_distill_features,
            photo_aug_features,
            temp,
        )
        loss_distill = add_infonce_distill(
            loss_distill,
            loss_dict,
            "infonce_sketch",
            getattr(args, "lambda_infonce_sketch", 0.0),
            sk_distill_features,
            sk_aug_features,
            temp,
        )
        lambda_text = getattr(args, "lambda_infonce_text", 0.0)
        if lambda_text > 0 and teacher_text_features is not None:
            loss_text = torch.tensor(0.0, device=pos_logits.device)
            loss_text = add_infonce_distill(
                loss_text,
                loss_dict,
                "infonce_photo_text",
                1.0,
                photo_text_distill_features,
                teacher_text_features,
                temp,
            )
            loss_text = add_infonce_distill(
                loss_text,
                loss_dict,
                "infonce_sketch_text",
                1.0,
                sk_text_distill_features,
                teacher_text_features,
                temp,
            )
            loss_distill = loss_distill + lambda_text * loss_text
    elif distill_mode == "teacher_weighted_ntxent":
        weight = getattr(args, "lambda_tw_ntxent", 0.0)
        if weight > 0:
            loss_value = teacher_weighted_ntxent_loss(
                sk_distill_features,
                photo_distill_features,
                sk_aug_features,
                photo_aug_features,
                alpha=getattr(args, "tw_alpha", 0.3),
                temperature=getattr(args, "tw_temperature", 0.08),
            )
            if loss_value is not None:
                loss_distill = loss_distill + weight * loss_value
                loss_dict["tw_ntxent"] = loss_value
    elif distill_mode == "independent_cosine":
        loss_distill = add_independent_cosine(
            loss_distill,
            loss_dict,
            "ind_photo",
            getattr(args, "lambda_ind_photo", 0.0),
            photo_distill_features,
            photo_aug_features,
        )
        loss_distill = add_independent_cosine(
            loss_distill,
            loss_dict,
            "ind_sketch",
            getattr(args, "lambda_ind_sketch", 0.0),
            sk_distill_features,
            sk_aug_features,
        )
        loss_distill = add_independent_cosine(
            loss_distill,
            loss_dict,
            "ind_sketch_photo",
            getattr(args, "lambda_ind_sketch_photo", 0.0),
            sk_distill_features,
            photo_aug_features,
        )
        loss_distill = add_independent_cosine(
            loss_distill,
            loss_dict,
            "ind_text",
            getattr(args, "lambda_ind_text", 0.0),
            0.5 * (photo_text_distill_features + sk_text_distill_features),
            teacher_text_features,
        )
    else:
        raise ValueError(f"Unknown distill_mode: {distill_mode}")

    lambda_adapter_rank = getattr(args, "lambda_adapter_rank", 0.0)
    if lambda_adapter_rank > 0:
        required = (
            photo_aug_features,
            sk_aug_features,
            photo_teacher_base,
            sketch_teacher_base,
        )
        if any(value is None for value in required):
            raise RuntimeError(
                "Adapter ranking requires adapted and base teacher image features."
            )
        rank_loss = adapter_guided_ranking_loss(
            sk_features,
            photo_features,
            sk_aug_features,
            photo_aug_features,
            sketch_teacher_base,
            photo_teacher_base,
            label,
            temperature=getattr(args, "rkd_temperature", 0.07),
            margin=getattr(args, "adapter_rank_margin", 0.2),
        )
        loss_distill = loss_distill + lambda_adapter_rank * rank_loss
        loss_dict["rank_adapter"] = rank_loss

    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, margin=0.2)
    loss_triplet = triplet(sk_feature_norm, photo_features_norm, neg_features)
    
    lambda_cls = getattr(args, "lambda_cls", 1.0)
    lambda_triplet = getattr(args, "lambda_triplet", 1.0)
    lambda_nt_xent = getattr(args, "lambda_nt_xent", 1.0)
    if lambda_nt_xent > 0:
        nt_xent_loss = nt_xent(photo_features, sk_features)
    else:
        nt_xent_loss = torch.tensor(0.0, device=pos_logits.device)
    
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
