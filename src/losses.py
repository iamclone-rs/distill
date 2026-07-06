"""Losses for the single supported DFN5B sketch-to-photo KD pipeline."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainingFeatures:
    photo_normalized: torch.Tensor
    sketch_normalized: torch.Tensor
    negative_normalized: torch.Tensor
    photo_logits: torch.Tensor
    sketch_logits: torch.Tensor
    student_photo: torch.Tensor
    student_sketch: torch.Tensor
    teacher_photo: torch.Tensor
    teacher_sketch: torch.Tensor
    labels: torch.Tensor


def kd_div_loss(
    student_sketch: torch.Tensor,
    student_photo: torch.Tensor,
    teacher_sketch: torch.Tensor,
    teacher_photo: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """KL(P_teacher || P_student) over row-wise sketch-to-photo similarities."""
    if temperature <= 0:
        raise ValueError("KD temperature must be positive.")

    target_device = student_sketch.device
    student_sketch = F.normalize(student_sketch.float(), dim=-1)
    student_photo = F.normalize(
        student_photo.to(device=target_device, dtype=torch.float32), dim=-1
    )
    teacher_sketch = F.normalize(
        teacher_sketch.to(device=target_device, dtype=torch.float32), dim=-1
    )
    teacher_photo = F.normalize(
        teacher_photo.to(device=target_device, dtype=torch.float32), dim=-1
    )

    student_log_prob = F.log_softmax(
        (student_sketch @ student_photo.t()) / temperature,
        dim=-1,
    )
    with torch.no_grad():
        teacher_prob = F.softmax(
            (teacher_sketch @ teacher_photo.t()) / temperature,
            dim=-1,
        )
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def compute_training_loss(
    features: TrainingFeatures,
    *,
    lambda_cls: float,
    lambda_triplet: float,
    lambda_kd: float,
    kd_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = features.labels.to(features.photo_logits.device)
    classification = F.cross_entropy(features.photo_logits, labels)
    classification = classification + F.cross_entropy(features.sketch_logits, labels)

    distance = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=distance,
        margin=0.2,
    )(
        features.sketch_normalized,
        features.photo_normalized,
        features.negative_normalized,
    )
    kd_loss = kd_div_loss(
        features.student_sketch,
        features.student_photo,
        features.teacher_sketch,
        features.teacher_photo,
        temperature=kd_temperature,
    )

    total = (
        lambda_cls * classification
        + lambda_triplet * triplet_loss
        + lambda_kd * kd_loss
    )
    return total, {
        "cls": classification,
        "triplet": triplet_loss,
        "kd_sk_ph": kd_loss,
    }
