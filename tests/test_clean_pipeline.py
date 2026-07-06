import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.finetune_teacher_adapter import (
    parse_args as parse_adapter_args,
    save_checkpoint,
)
from src.losses import TrainingFeatures, compute_training_loss
from src.main_train import parse_args as parse_student_args
from src.model import CustomCLIP
from src.sketchy_dataset import TrainDataset, ValidDataset
from src.teacher_adapter import (
    DFN_MODEL_NAME,
    DFN_PRETRAINED,
    ModalityAdapters,
    ResidualAdapter,
    load_adapter_checkpoint,
)


class TinyVisual(nn.Module):
    input_resolution = 224
    output_dim = 8
    image_size = 224

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.output_dim)

    def forward(self, images, visual_context=None, deep_prompts=None):
        return self.projection(images.mean(dim=(-1, -2)))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([nn.Identity()])


class TinyStudentCLIP(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.visual = TinyVisual()
        self.transformer = TinyTransformer()
        self.token_embedding = nn.Embedding(49_408, 16)
        self.positional_embedding = nn.Parameter(torch.randn(77, 16) * 0.01)
        self.ln_final = nn.LayerNorm(16)
        self.text_projection = nn.Parameter(torch.randn(16, 8) * 0.01)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = SimpleNamespace(image_size=224, output_dim=12)
        self.projection = nn.Linear(3, 12)

    def encode_image(self, images):
        return self.projection(images.mean(dim=(-1, -2)))


class CleanPipelineTests(unittest.TestCase):
    def test_residual_adapter_starts_as_identity(self):
        torch.manual_seed(1)
        inputs = F.normalize(torch.randn(6, 32), dim=-1)
        outputs = ResidualAdapter(32, 4)(inputs)
        torch.testing.assert_close(outputs, inputs)

    def test_legacy_residual_checkpoint_loads(self):
        adapters = ModalityAdapters(32, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save(
                {
                    "adapter_state_dict": adapters.state_dict(),
                    "feature_dim": 32,
                    "bottleneck_dim": 4,
                    "model": DFN_MODEL_NAME,
                    "pretrained": DFN_PRETRAINED,
                },
                path,
            )
            loaded, checkpoint = load_adapter_checkpoint(
                path, expected_feature_dim=32
            )
        self.assertEqual(checkpoint["bottleneck_dim"], 4)
        for expected, actual in zip(adapters.parameters(), loaded.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_non_residual_checkpoint_is_rejected(self):
        adapters = ModalityAdapters(16, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "non_residual.pt"
            torch.save(
                {
                    "adapter_state_dict": adapters.state_dict(),
                    "feature_dim": 16,
                    "bottleneck_dim": 2,
                    "adapter_mode": "non_residual",
                },
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "only supports residual"):
                load_adapter_checkpoint(path)

    def test_new_checkpoint_roundtrip(self):
        adapters = ModalityAdapters(24, 3)
        args = SimpleNamespace(
            bottleneck_dim=3,
            dataset="sketchy_1",
            root="unused",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, adapters, args, 1, {}, ["seen"], ["unseen"])
            loaded, checkpoint = load_adapter_checkpoint(
                path, expected_feature_dim=24
            )
        self.assertEqual(checkpoint["model"], DFN_MODEL_NAME)
        self.assertEqual(checkpoint["pretrained"], DFN_PRETRAINED)
        self.assertEqual(sum(p.numel() for p in loaded.parameters()), 438)

    def test_clean_clis_reject_removed_teacher_flags(self):
        adapter_args = parse_adapter_args(
            [
                "--root",
                "data",
                "--dataset",
                "sketchy_1",
                "--output_dir",
                "runs/test",
            ]
        )
        self.assertEqual(adapter_args.bottleneck_dim, 32)
        student_args = parse_student_args(
            [
                "--root",
                "data",
                "--dataset",
                "sketchy_1",
                "--teacher_adapter_ckpt",
                "best.pt",
            ]
        )
        self.assertEqual(student_args.lambda_kd, 2.5)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_student_args(
                    [
                        "--root",
                        "data",
                        "--dataset",
                        "sketchy_1",
                        "--teacher_adapter_ckpt",
                        "best.pt",
                        "--teacher",
                        "dfn5b",
                    ]
                )

    def test_clean_loss_matches_previous_active_formula(self):
        torch.manual_seed(7)
        batch_size = 5
        features = TrainingFeatures(
            photo_normalized=F.normalize(torch.randn(batch_size, 8), dim=-1),
            sketch_normalized=F.normalize(torch.randn(batch_size, 8), dim=-1),
            negative_normalized=F.normalize(torch.randn(batch_size, 8), dim=-1),
            photo_logits=torch.randn(batch_size, 6),
            sketch_logits=torch.randn(batch_size, 6),
            student_photo=torch.randn(batch_size, 8),
            student_sketch=torch.randn(batch_size, 8),
            teacher_photo=torch.randn(batch_size, 12),
            teacher_sketch=torch.randn(batch_size, 12),
            labels=torch.arange(batch_size),
        )
        actual, parts = compute_training_loss(
            features,
            lambda_cls=1.0,
            lambda_triplet=1.0,
            lambda_kd=2.5,
            kd_temperature=0.07,
        )

        classification = F.cross_entropy(features.photo_logits, features.labels)
        classification += F.cross_entropy(features.sketch_logits, features.labels)
        triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=0.2,
        )(
            features.sketch_normalized,
            features.photo_normalized,
            features.negative_normalized,
        )
        student_logits = (
            F.normalize(features.student_sketch.float(), dim=-1)
            @ F.normalize(features.student_photo.float(), dim=-1).t()
        ) / 0.07
        teacher_logits = (
            F.normalize(features.teacher_sketch.float(), dim=-1)
            @ F.normalize(features.teacher_photo.float(), dim=-1).t()
        ) / 0.07
        kd = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        expected = classification + triplet + 2.5 * kd
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(parts["kd_sk_ph"], kd)

    def test_all_dataset_keys_have_sorted_samples(self):
        from src.data_config import UNSEEN_CLASSES

        for dataset in tuple(UNSEEN_CLASSES):
            original = UNSEEN_CLASSES[dataset]
            try:
                UNSEEN_CLASSES[dataset] = ["unseen"]
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    for modality in ("sketch", "photo"):
                        for category in ("seen_b", "unseen", "seen_a"):
                            target = root / modality / category
                            target.mkdir(parents=True)
                            Image.new("RGB", (8, 8), "white").save(target / "1.png")
                    train = TrainDataset(root, dataset)
                    validation = ValidDataset(root, dataset, modality="sketch")
                    self.assertEqual(train.categories, ["seen_a", "seen_b"])
                    self.assertEqual(len(validation), 1)
                    sample = train[0]
                    self.assertEqual(len(sample), 6)
                    self.assertIn(sample[-1], (0, 1))
            finally:
                UNSEEN_CLASSES[dataset] = original

    def test_custom_clip_smoke_forward_backward(self):
        torch.manual_seed(11)
        teacher = TinyTeacher().eval().requires_grad_(False)
        adapters = ModalityAdapters(12, 3).eval().requires_grad_(False)
        model = CustomCLIP(
            SimpleNamespace(n_ctx=1),
            TinyStudentCLIP(),
            TinyStudentCLIP(),
            teacher,
            adapters,
        )
        model.train()
        self.assertFalse(teacher.training)
        self.assertFalse(adapters.training)
        batch = (
            torch.randn(2, 3, 224, 224),
            torch.randn(2, 3, 224, 224),
            torch.randn(2, 3, 224, 224),
            torch.randn(2, 3, 224, 224),
            torch.randn(2, 3, 224, 224),
            torch.tensor([0, 1]),
        )
        features = model(batch, ["seen_a", "seen_b"])
        loss, _ = compute_training_loss(
            features,
            lambda_cls=1.0,
            lambda_triplet=1.0,
            lambda_kd=2.5,
            kd_temperature=0.07,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.prompt_learner_photo.ctx.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
