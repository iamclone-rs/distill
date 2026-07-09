"""Losses used by the DFN5B-to-CoPrompt SBIR benchmark."""

import torch
import torch.nn as nn
from torch.nn import functional as F


def relational_kd_loss(
    student_sketch,
    student_photo,
    teacher_sketch,
    teacher_photo,
    temperature=0.07,
):
    """Match the student and teacher sketch-to-photo similarity distributions."""
    student_device = student_sketch.device
    student_sketch = F.normalize(student_sketch.float(), dim=-1)
    student_photo = F.normalize(student_photo.float(), dim=-1)
    teacher_sketch = F.normalize(
        teacher_sketch.to(device=student_device, dtype=torch.float32), dim=-1
    )
    teacher_photo = F.normalize(
        teacher_photo.to(device=student_device, dtype=torch.float32), dim=-1
    )

    student_logits = student_sketch @ student_photo.t() / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    with torch.no_grad():
        teacher_logits = teacher_sketch @ teacher_photo.t() / temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def image_text_kd_loss(
    student_image,
    student_text,
    teacher_image,
    teacher_text,
    temperature=0.07,
):
    """Match teacher and student image-to-text class distributions."""
    student_device = student_image.device
    student_image = F.normalize(student_image.float(), dim=-1)
    student_text = F.normalize(student_text.float(), dim=-1)
    teacher_image = F.normalize(
        teacher_image.to(device=student_device, dtype=torch.float32), dim=-1
    )
    teacher_text = F.normalize(
        teacher_text.to(device=student_device, dtype=torch.float32), dim=-1
    )

    student_logits = student_image @ student_text.t() / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    with torch.no_grad():
        teacher_logits = teacher_image @ teacher_text.t() / temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
        temperature ** 2
    )


def image_text_kd_total(
    sketch_features,
    photo_features,
    student_sketch_text,
    student_photo_text,
    teacher_sketch_features,
    teacher_photo_features,
    teacher_sketch_text,
    teacher_photo_text,
    mode,
    temperature,
):
    losses = []
    if mode in ("sketch", "both"):
        losses.append(
            image_text_kd_loss(
                sketch_features,
                student_sketch_text,
                teacher_sketch_features,
                teacher_sketch_text,
                temperature,
            )
        )
    if mode in ("photo", "both"):
        losses.append(
            image_text_kd_loss(
                photo_features,
                student_photo_text,
                teacher_photo_features,
                teacher_photo_text,
                temperature,
            )
        )
    if not losses:
        return sketch_features.new_zeros(())
    return torch.stack(losses).mean()


def batch_hard_teacher_triplet_loss(
    sketch_features,
    photo_features,
    labels,
    margin=0.2,
):
    """Symmetric batch-hard triplet loss for the jointly trained teacher adapters."""
    sketch_features = F.normalize(sketch_features.float(), dim=-1)
    photo_features = F.normalize(photo_features.float(), dim=-1)
    labels = labels.to(sketch_features.device)

    distance = 1.0 - sketch_features @ photo_features.t()
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = ~positive_mask

    def one_direction(dist):
        valid_negative = negative_mask.any(dim=-1)
        hardest_positive = dist.masked_fill(~positive_mask, -torch.inf).max(dim=-1).values
        hardest_negative = dist.masked_fill(~negative_mask, torch.inf).min(dim=-1).values
        losses = F.relu(hardest_positive - hardest_negative + margin)
        if valid_negative.any():
            return losses[valid_negative].mean()
        return dist.new_zeros(())

    return 0.5 * (one_direction(distance) + one_direction(distance.t()))


def teacher_semantic_loss(
    sketch_features,
    photo_features,
    labels,
    sketch_text,
    photo_text,
    temperature,
):
    return 0.5 * (
        F.cross_entropy(sketch_features @ sketch_text.t() / temperature, labels)
        + F.cross_entropy(photo_features @ photo_text.t() / temperature, labels)
    )


def loss_fn(args, features):
    (
        photo_features,
        sketch_features,
        teacher_photo_features,
        teacher_sketch_features,
        negative_features,
        labels,
        photo_logits,
        sketch_logits,
        teacher_active,
        joint_teacher_adapter,
        teacher_sketch_text,
        teacher_photo_text,
        student_sketch_text,
        student_photo_text,
    ) = features

    labels = labels.to(photo_logits.device)
    classification_loss = (
        F.cross_entropy(photo_logits, labels)
        + F.cross_entropy(sketch_logits, labels)
    )

    cosine_distance = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=cosine_distance,
        margin=0.2,
    )(sketch_features, photo_features, negative_features)

    kd_loss = torch.zeros((), device=photo_logits.device)
    if teacher_active and args.lambda_kd > 0:
        kd_loss = relational_kd_loss(
            sketch_features,
            photo_features,
            teacher_sketch_features,
            teacher_photo_features,
            args.kd_temperature,
        )

    image_text_kd = torch.zeros((), device=photo_logits.device)
    if (
        teacher_active
        and args.lambda_image_text_kd > 0
        and args.image_text_kd_mode != "none"
        and teacher_sketch_text is not None
        and teacher_photo_text is not None
    ):
        image_text_kd = image_text_kd_total(
            sketch_features,
            photo_features,
            student_sketch_text,
            student_photo_text,
            teacher_sketch_features,
            teacher_photo_features,
            teacher_sketch_text,
            teacher_photo_text,
            args.image_text_kd_mode,
            args.image_text_kd_temperature,
        )

    teacher_triplet_loss = torch.zeros((), device=photo_logits.device)
    teacher_semantic = torch.zeros((), device=photo_logits.device)
    if joint_teacher_adapter:
        teacher_triplet_loss = batch_hard_teacher_triplet_loss(
            teacher_sketch_features,
            teacher_photo_features,
            labels,
            args.teacher_triplet_margin,
        )
        teacher_semantic = teacher_semantic_loss(
            teacher_sketch_features,
            teacher_photo_features,
            labels,
            teacher_sketch_text,
            teacher_photo_text,
            args.teacher_temperature,
        )

    total_loss = (
        args.lambda_cls * classification_loss
        + args.lambda_triplet * triplet_loss
        + args.lambda_kd * kd_loss
        + args.lambda_image_text_kd * image_text_kd
        + args.lambda_teacher_retrieval * teacher_triplet_loss
        + args.lambda_teacher_semantic * teacher_semantic
    )
    return total_loss, {
        "cls": classification_loss,
        "triplet": triplet_loss,
        "kd_sketch_photo": kd_loss,
        "image_text_kd": image_text_kd,
        "teacher_triplet": teacher_triplet_loss,
        "teacher_semantic": teacher_semantic,
    }
