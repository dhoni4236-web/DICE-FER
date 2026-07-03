from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from PIL import Image
except ImportError as exc:
    Image = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

try:
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    DataLoader = None

    class Dataset:
        pass


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


# Named factory required for pickle on Windows multiprocessing (no lambdas)
def _identity_dict_factory():
    return defaultdict(list)


@dataclass(frozen=True)
class FERSample:
    image_path: Path
    expression: str
    identity: str


class FERPairDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[FERSample],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        return_paths: bool = True,
        image_mode: str = "RGB",
    ) -> None:
        self.samples          = list(samples)
        self.transform        = transform
        self.target_transform = target_transform
        self.seed             = seed
        self.return_paths     = return_paths
        self.image_mode       = image_mode
        self.epoch            = 0
        self._rng             = random.Random(seed)

        if not self.samples:
            raise ValueError("FERPairDataset needs at least one sample.")

        # Build expression → identity → [indices] lookup
        self.by_expression_identity: dict[str, dict[str, list[int]]] = defaultdict(
            _identity_dict_factory
        )
        for idx, sample in enumerate(self.samples):
            if not sample.expression:
                raise ValueError(f"Sample at index {idx} has an empty expression.")
            if not sample.identity:
                raise ValueError(f"Sample at index {idx} has an empty identity.")
            self.by_expression_identity[sample.expression][sample.identity].append(idx)

        self.valid_anchor_indices = self._build_valid_anchor_indices()
        if not self.valid_anchor_indices:
            raise ValueError(
                "No valid FER pairs found. Each usable expression must contain "
                "images from at least two different identities."
            )

        # ── FIX perf: pre-compute per-anchor partner candidate lists ──────────
        # _sample_partner_index previously rebuilt `candidate_identities` on every
        # __getitem__ call by iterating the dict and filtering. With augmented
        # datasets (160K+ samples) this is O(N_identities) per item, every epoch.
        # Pre-computing saves millions of list compreh. calls per training run.
        self._partner_candidates: dict[int, list[list[int]]] = {}
        for anchor_idx in self.valid_anchor_indices:
            anchor = self.samples[anchor_idx]
            id_map = self.by_expression_identity[anchor.expression]
            # List of index-lists, one per different identity
            candidates = [
                indices
                for identity, indices in id_map.items()
                if identity != anchor.identity
            ]
            self._partner_candidates[anchor_idx] = candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Class-method constructors
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        path_col: str = "image_path",
        expression_col: str = "expression",
        identity_col: str = "identity",
        **kwargs,
    ) -> "FERPairDataset":
        csv_path = Path(csv_path)
        root     = Path(image_root) if image_root is not None else csv_path.parent
        samples: list[FERSample] = []
        with csv_path.open(newline="") as handle:
            reader  = csv.DictReader(handle)
            missing = {path_col, expression_col, identity_col}.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
            for row in reader:
                raw_path   = row[path_col].strip()
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = root / image_path
                samples.append(
                    FERSample(
                        image_path=image_path,
                        expression=row[expression_col].strip(),
                        identity=row[identity_col].strip(),
                    )
                )
        return cls(samples, **kwargs)

    @classmethod
    def from_image_folder(
        cls,
        root: str | Path,
        identity_from: str | Callable[[Path, Path, str], str] = "filename_prefix",
        **kwargs,
    ) -> "FERPairDataset":
        root    = Path(root)
        samples: list[FERSample] = []
        for image_path in sorted(root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = image_path.relative_to(root)
            if len(rel.parts) < 2:
                continue
            expression = rel.parts[0]
            identity   = _infer_identity(image_path, root, expression, identity_from)
            samples.append(FERSample(image_path=image_path, expression=expression, identity=identity))
        return cls(samples, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ─────────────────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to vary pair sampling across epochs."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.valid_anchor_indices)

    def __getitem__(self, item_index: int) -> dict:
        anchor_index  = self.valid_anchor_indices[item_index]
        partner_index = self._sample_partner_index(anchor_index)

        anchor  = self.samples[anchor_index]
        partner = self.samples[partner_index]

        image_m = self._load_image(anchor.image_path)
        image_n = self._load_image(partner.image_path)

        if self.transform is not None:
            image_m = self.transform(image_m)
            image_n = self.transform(image_n)

        expression = anchor.expression
        if self.target_transform is not None:
            expression = self.target_transform(expression)

        item: dict = {
            "image_m":    image_m,
            "image_n":    image_n,
            "expression": expression,
            "identity_m": anchor.identity,
            "identity_n": partner.identity,
            "index_m":    anchor_index,
            "index_n":    partner_index,
        }
        if self.return_paths:
            item["path_m"] = str(anchor.image_path)
            item["path_n"] = str(partner.image_path)
        return item

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_valid_anchor_indices(self) -> list[int]:
        valid = []
        for expression, identity_to_indices in self.by_expression_identity.items():
            if len(identity_to_indices) < 2:
                continue  # expression has only one identity — cannot form a pair
            for indices in identity_to_indices.values():
                valid.extend(indices)
        return valid

    def _sample_partner_index(self, anchor_index: int) -> int:
        # ── FIX perf: use pre-computed candidate lists, no per-call filtering ─
        candidate_groups = self._partner_candidates[anchor_index]
        if not candidate_groups:
            raise RuntimeError(
                f"No different-identity partner for sample index {anchor_index} "
                f"(expression={self.samples[anchor_index].expression!r})."
            )
        rng            = self._rng_for_anchor(anchor_index)
        partner_group  = rng.choice(candidate_groups)   # pick a different-identity group
        return rng.choice(partner_group)                 # pick one image from that group

    def _rng_for_anchor(self, anchor_index: int) -> random.Random:
        # ── FIX perf: reuse module-level _rng when seed is None ──────────────
        # When seed is set, create a lightweight deterministic RNG so pair
        # sampling varies across epochs but is reproducible.
        # Previously this constructed a new random.Random() on every call,
        # which is expensive (millions of calls per epoch on large datasets).
        if self.seed is None:
            return self._rng
        # Combine seed + epoch + anchor to get a unique but reproducible state.
        # Using a single integer seed avoids the Random() constructor overhead
        # by reseeding a shared instance — but that's not thread-safe.
        # The constructor call is unavoidable for correctness; we keep it here
        # but avoid all other per-call allocations by using pre-computed candidates.
        return random.Random(self.seed + self.epoch * 1_000_003 + anchor_index)

    def _load_image(self, image_path: Path):
        if Image is None:
            raise ImportError("Pillow is required to load images.") from _PIL_IMPORT_ERROR
        # Keep file handle open only as long as needed; convert() copies pixel
        # data so closing the handle afterward is safe.
        with Image.open(image_path) as image:
            return image.convert(self.image_mode)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def create_pair_dataloader(
    dataset: FERPairDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    persistent_workers: bool | None = None,
    **kwargs,
) -> "DataLoader":
    """
    Create a DataLoader for FERPairDataset.

    persistent_workers defaults to True when num_workers > 0, which avoids
    re-spawning worker processes between epochs (significant overhead on large
    datasets). Pass persistent_workers=False to disable if memory is tight.
    """
    if DataLoader is None:
        raise ImportError("PyTorch is required to create a DataLoader.")

    # ── FIX perf: persistent_workers avoids per-epoch worker re-spawn ────────
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Identity inference helper
# ─────────────────────────────────────────────────────────────────────────────

def _infer_identity(
    image_path: Path,
    root: Path,
    expression: str,
    identity_from: str | Callable[[Path, Path, str], str],
) -> str:
    if callable(identity_from):
        return str(identity_from(image_path, root, expression))

    if identity_from == "parent":
        rel = image_path.relative_to(root)
        if len(rel.parts) < 3:
            raise ValueError(
                "identity_from='parent' expects root/expression/identity/image files."
            )
        return rel.parts[1]

    if identity_from == "filename_prefix":
        return image_path.stem.split("_", 1)[0]

    raise ValueError(
        "identity_from must be 'filename_prefix', 'parent', or a callable."
    )
