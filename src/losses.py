import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def rkd_loss(student_feat1, student_feat2, teacher_feat1, teacher_feat2, temperature=0.07):
    """
    Relational Knowledge Distillation sử dụng KL-Divergence trên cosine similarity matrix.
    Ép phân bố tương quan giữa (feat1, feat2) của Student phải giống với của Teacher.
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

    if getattr(args, "use_rkd", False):
        temp = getattr(args, "rkd_temperature", 0.07)
        loss_image_distill = torch.tensor(0.0, device=pos_logits.device)
        loss_text_distill = torch.tensor(0.0, device=pos_logits.device)
        lambda_text_distill = 1.0  # RKD đã nhân hệ số lambda_rkd_* vào bên trong loss_text_distill, nên ở bước cộng tổng cuối cùng chỉ nhân với 1.0

        # 1. RKD Sketch-Photo (Thay thế InfoNCE Photo/Sketch Distill cũ)
        lambda_rkd_sk_ph = getattr(args, "lambda_rkd_sk_ph", 0.0)
        if lambda_rkd_sk_ph > 0:
            rkd_sk_ph = rkd_loss(sk_distill_features, photo_distill_features, sk_aug_features, photo_aug_features, temp)
            loss_image_distill = loss_image_distill + lambda_rkd_sk_ph * rkd_sk_ph
            loss_dict['rkd_sk_ph'] = rkd_sk_ph

        # 2. RKD Photo-Text và Sketch-Text (Hỗ trợ Zero-Shot)
        lambda_rkd_ph_txt = getattr(args, "lambda_rkd_ph_txt", 0.0)
        lambda_rkd_sk_txt = getattr(args, "lambda_rkd_sk_txt", 0.0)

        if (lambda_rkd_ph_txt > 0 or lambda_rkd_sk_txt > 0) and teacher_text_features is not None:
            if lambda_rkd_ph_txt > 0 and photo_text_distill_features is not None:
                rkd_ph_txt = rkd_loss(photo_distill_features, photo_text_distill_features, photo_aug_features, teacher_text_features, temp)
                loss_text_distill = loss_text_distill + lambda_rkd_ph_txt * rkd_ph_txt
                loss_dict['rkd_ph_txt'] = rkd_ph_txt
                
            if lambda_rkd_sk_txt > 0 and sk_text_distill_features is not None:
                rkd_sk_txt = rkd_loss(sk_distill_features, sk_text_distill_features, sk_aug_features, teacher_text_features, temp)
                loss_text_distill = loss_text_distill + lambda_rkd_sk_txt * rkd_sk_txt
                loss_dict['rkd_sk_txt'] = rkd_sk_txt
    else:
        lambda_photo_distill = getattr(args, "lambda_photo_distill", 0.0)
        lambda_sketch_distill = getattr(args, "lambda_sketch_distill", 0.0)
        lambda_text_distill = getattr(args, "lambda_text_distill", 0.0)
        loss_image_distill = torch.tensor(0.0, device=pos_logits.device)
    
        if lambda_photo_distill > 0:
            loss_distill_photo = cross_loss(photo_distill_features, photo_aug_features, args)
            loss_image_distill = loss_image_distill + lambda_photo_distill * loss_distill_photo
            loss_dict['distill_photo'] = loss_distill_photo
    
        if lambda_sketch_distill > 0:
            loss_distill_sk = cross_loss(sk_distill_features, sk_aug_features, args)
            loss_image_distill = loss_image_distill + lambda_sketch_distill * loss_distill_sk
            loss_dict['distill_sk'] = loss_distill_sk
    
        loss_text_distill = torch.tensor(0.0, device=pos_logits.device)
        if (
            lambda_text_distill > 0
            and teacher_text_features is not None
            and photo_text_distill_features is not None
            and sk_text_distill_features is not None
        ):
            loss_text_distill = (
                cross_loss(photo_text_distill_features, teacher_text_features, args)
                + cross_loss(sk_text_distill_features, teacher_text_features, args)
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
        + loss_image_distill \
        + lambda_text_distill * loss_text_distill \
        + lambda_nt_xent * nt_xent_loss
    )
    
    loss_dict['cls'] = loss_cls
    loss_dict['triplet'] = loss_triplet
    loss_dict['nt_xent'] = nt_xent_loss
    
    return total_loss, loss_dict
