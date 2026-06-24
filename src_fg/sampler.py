import os
import numpy as np
from torch.utils.data import Sampler

class EqualLabelSampler(Sampler):
    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        samples_per_class: int,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size % samples_per_class != 0:
            raise ValueError("batch_size must be divisible by samples_per_class")

        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class
        self.seed = seed
        self.drop_last = drop_last

        cat_to_label = {c: i for i, c in enumerate(self.dataset.all_categories)}
        label_to_indices = {}
        for idx, sk_path in enumerate(self.dataset.all_sketches_path):
            category = sk_path.split(os.path.sep)[-2]
            lab = cat_to_label.get(category, None)
            if lab is None:
                continue
            label_to_indices.setdefault(lab, []).append(idx)

        self.label_to_indices = {k: v for k, v in label_to_indices.items() if len(v) > 0}
        self.labels = list(self.label_to_indices.keys())
        if len(self.labels) == 0:
            raise ValueError("No labels found for PK sampling")

        self._epoch = 0

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(len(self)):
            chosen_labels = rng.choice(
                self.labels,
                size=self.classes_per_batch,
                replace=len(self.labels) < self.classes_per_batch,
            )
            batch = []
            for lab in chosen_labels:
                pool = self.label_to_indices[lab]
                replace = len(pool) < self.samples_per_class
                picked = rng.choice(pool, size=self.samples_per_class, replace=replace)
                batch.extend(int(i) for i in picked.tolist())
            yield batch