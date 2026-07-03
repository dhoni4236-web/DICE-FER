from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from .losses import (
    discriminator_loss,
    encoder_adversarial_loss,
    estimate_global_mi,
    estimate_local_mi,
)
from .model import DICEFER


@dataclass
class DICEFERConfig:
    mu_exp: float = 0.5
    nu_exp: float = 1.0
    mu_id: float = 0.5
    nu_id: float = 1.0
    delta: float = 0.1
    zeta_adv: float = 0.01
    learning_rate: float = 1e-4
    classifier_learning_rate: float = 1e-4

    # ── FIX Bug 1 (partial): gradient clipping for identity MI stability ──────
    # DV bound is unbounded above; without clipping the identity MI diverges.
    grad_clip_norm: float = 10.0

    # Early stopping for identity stage: stop if MI stops improving
    identity_patience: int = 10

    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


class DICEFERTrainer:
    def __init__(self, model: DICEFER, config: DICEFERConfig) -> None:
        self.model  = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        # ── Expression stage: both encoders + all 4 stats nets (M and N) ──────
        # FIX Bug 5: now includes separate _m and _n stats nets
        self.expression_optimizer = torch.optim.Adam(
            list(model.expression_encoder.parameters())
            + list(model.exp_global_stats_m.parameters())
            + list(model.exp_global_stats_n.parameters())
            + list(model.exp_local_stats_m.parameters())
            + list(model.exp_local_stats_n.parameters()),
            lr=config.learning_rate,
        )

        # ── Identity stage: identity encoder + identity stats nets ────────────
        self.identity_optimizer = torch.optim.Adam(
            list(model.identity_encoder.parameters())
            + list(model.id_global_stats_m.parameters())
            + list(model.id_global_stats_n.parameters())
            + list(model.id_local_stats_m.parameters())
            + list(model.id_local_stats_n.parameters()),
            lr=config.learning_rate,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=config.learning_rate,
        )
        self.classifier_optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=config.classifier_learning_rate,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # EXPRESSION STAGE
    # ─────────────────────────────────────────────────────────────────────────
    def train_expression_epoch(self, loader: Iterable, epoch: int) -> dict[str, float]:
        self.model.train()
        metrics = MetricTracker()
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)

        for batch in tqdm(loader, desc=f"expression epoch {epoch}", leave=False):
            image_m, image_n, _ = self._unpack_batch(batch)

            exp_m = self.model.encode_expression(image_m)
            exp_n = self.model.encode_expression(image_n)

            # ── FIX Bug 5: use separate stats nets for M→N and N→M directions ─
            mi = self.config.mu_exp * (
                estimate_global_mi(self.model.exp_global_stats_m, exp_m.global_features, exp_n.embedding)
                + estimate_global_mi(self.model.exp_global_stats_n, exp_n.global_features, exp_m.embedding)
            ) + self.config.nu_exp * (
                estimate_local_mi(self.model.exp_local_stats_m, exp_m.local_features, exp_n.embedding)
                + estimate_local_mi(self.model.exp_local_stats_n, exp_n.local_features, exp_m.embedding)
            )

            l1   = F.l1_loss(exp_m.embedding, exp_n.embedding)
            loss = -mi + self.config.delta * l1

            self.expression_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # ── FIX Bug 1 (expression stage): clip gradients ─────────────────
            nn.utils.clip_grad_norm_(
                [p for g in self.expression_optimizer.param_groups for p in g["params"]],
                self.config.grad_clip_norm,
            )
            self.expression_optimizer.step()

            metrics.update(loss=loss, expression_mi=mi, expression_l1=l1)
        return metrics.compute()

    # ─────────────────────────────────────────────────────────────────────────
    # IDENTITY STAGE
    # ─────────────────────────────────────────────────────────────────────────
    def train_identity_epoch(self, loader: Iterable, epoch: int) -> dict[str, float]:
        self.model.train()

        # Freeze expression encoder — identity stage must not touch it
        self.model.expression_encoder.eval()
        for param in self.model.expression_encoder.parameters():
            param.requires_grad_(False)

        metrics = MetricTracker()
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)

        for batch in tqdm(loader, desc=f"identity epoch {epoch}", leave=False):
            image_m, image_n, _ = self._unpack_batch(batch)

            # Get frozen expression embeddings (no grad)
            with torch.no_grad():
                e_m = self.model.encode_expression(image_m).embedding
                e_n = self.model.encode_expression(image_n).embedding

            # ── Step 1: train discriminator ───────────────────────────────────
            # Use fresh identity embeddings (detached) for discriminator update.
            # FIX Bug 2+3: discriminator_loss now uses correct real/fake labels.
            i_m_det = self.model.encode_identity(image_m).embedding.detach()
            i_n_det = self.model.encode_identity(image_n).embedding.detach()

            disc_loss = (
                discriminator_loss(self.model.discriminator, e_m, i_m_det)
                + discriminator_loss(self.model.discriminator, e_n, i_n_det)
            )
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            disc_loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(),
                self.config.grad_clip_norm,
            )
            self.discriminator_optimizer.step()

            # ── Step 2: train identity encoder (MI max + adversarial min) ─────
            id_m = self.model.encode_identity(image_m)
            id_n = self.model.encode_identity(image_n)

            t_m = torch.cat([e_m, id_m.embedding], dim=1)   # [B, 128]
            t_n = torch.cat([e_n, id_n.embedding], dim=1)

            # FIX Bug 5: use separate M/N stats nets for identity MI too
            id_mi = self.config.mu_id * (
                estimate_global_mi(self.model.id_global_stats_m, id_m.global_features, t_m)
                + estimate_global_mi(self.model.id_global_stats_n, id_n.global_features, t_n)
            ) + self.config.nu_id * (
                estimate_local_mi(self.model.id_local_stats_m, id_m.local_features, t_m)
                + estimate_local_mi(self.model.id_local_stats_n, id_n.local_features, t_n)
            )

            # FIX Bug 3: encoder_adversarial_loss now targets label 0 (fool discriminator)
            adv = (
                encoder_adversarial_loss(self.model.discriminator, e_m, id_m.embedding)
                + encoder_adversarial_loss(self.model.discriminator, e_n, id_n.embedding)
            )

            loss = -id_mi + self.config.zeta_adv * adv

            self.identity_optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # ── FIX Bug 1: clip identity gradients to prevent MI explosion ────
            nn.utils.clip_grad_norm_(
                [p for g in self.identity_optimizer.param_groups for p in g["params"]],
                self.config.grad_clip_norm,
            )
            self.identity_optimizer.step()

            metrics.update(loss=loss, identity_mi=id_mi, adversarial=adv, discriminator=disc_loss)
        return metrics.compute()

    # ─────────────────────────────────────────────────────────────────────────
    # CLASSIFIER STAGE
    # ─────────────────────────────────────────────────────────────────────────
    def train_classifier_epoch(
        self,
        loader: Iterable,
        epoch: int,
        fine_tune_encoder: bool = False,
    ) -> dict[str, float]:
        self.model.train()
        if not fine_tune_encoder:
            self.model.expression_encoder.eval()

        metrics = MetricTracker()
        for batch in tqdm(loader, desc=f"classifier epoch {epoch}", leave=False):
            image_m, _, target = self._unpack_batch(batch)

            if fine_tune_encoder:
                expression = self.model.encode_expression(image_m).embedding
            else:
                with torch.no_grad():
                    expression = self.model.encode_expression(image_m).embedding

            logits = self.model.classifier(expression)
            loss   = F.cross_entropy(logits, target)

            self.classifier_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.classifier_optimizer.step()

            accuracy = (logits.argmax(dim=1) == target).float().mean()
            metrics.update(loss=loss, accuracy=accuracy)
        return metrics.compute()

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_classifier(
        self,
        loader: Iterable,
        class_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        prefix: str = "evaluation",
    ) -> dict[str, float]:
        self.model.eval()
        total_loss  = 0.0
        total_items = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        for batch in tqdm(loader, desc="evaluate", leave=False):
            image_m, _, target = self._unpack_batch(batch)
            logits      = self.model.classify_expression(image_m)
            loss        = F.cross_entropy(logits, target)
            predictions = logits.argmax(dim=1)

            batch_size   = target.numel()
            total_loss  += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            y_true.extend(target.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

        if total_items == 0:
            return {}

        labels       = list(range(len(class_names))) if class_names is not None else sorted(set(y_true) | set(y_pred))
        target_names = class_names if class_names is not None else [str(l) for l in labels]
        accuracy     = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0,
        )
        metrics = {
            "loss":               total_loss / total_items,
            "accuracy":           accuracy,
            "precision_macro":    float(precision_macro),
            "recall_macro":       float(recall_macro),
            "f1_macro":           float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted":    float(recall_weighted),
            "f1_weighted":        float(f1_weighted),
        }

        if output_dir is not None:
            report = classification_report(
                y_true, y_pred, labels=labels, target_names=target_names,
                output_dict=True, zero_division=0,
            )
            matrix = confusion_matrix(y_true, y_pred, labels=labels)
            self._save_evaluation_outputs(Path(output_dir), prefix, metrics, report, matrix, target_names)

        return metrics

    @torch.no_grad()
    def estimate_mig(self, loader: Iterable) -> float:
        self.model.eval()
        shared_mi_vals:  list[torch.Tensor] = []
        overlap_mi_vals: list[torch.Tensor] = []

        for batch in tqdm(loader, desc="estimate MIG", leave=False):
            image_m, image_n, _ = self._unpack_batch(batch)

            exp_m = self.model.encode_expression(image_m)
            exp_n = self.model.encode_expression(image_n)
            id_m  = self.model.encode_identity(image_m)

            shared_mi = estimate_global_mi(
                self.model.exp_global_stats_m,
                exp_m.global_features,
                exp_n.embedding,
            )
            t_m = torch.cat([exp_m.embedding, id_m.embedding], dim=1)
            overlap_mi = estimate_global_mi(
                self.model.id_global_stats_m,
                id_m.global_features,
                t_m,
            )
            shared_mi_vals.append(shared_mi.detach())
            overlap_mi_vals.append(overlap_mi.detach())

        mean_shared  = torch.stack(shared_mi_vals).mean().item()
        mean_overlap = torch.stack(overlap_mi_vals).mean().item()
        return float(mean_shared - mean_overlap)

    def save_checkpoint(self, path: str | Path, classes: list[str], epoch: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch":   epoch,
                "classes": classes,
                "model":   self.model.state_dict(),
                "config":  self.config.__dict__,
            },
            path,
        )

    def _unpack_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_m = batch["image_m"].to(self.device, non_blocking=True)
        image_n = batch["image_n"].to(self.device, non_blocking=True)
        target  = batch["expression"]
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.long)
        return image_m, image_n, target.to(self.device, non_blocking=True).long()

    def _save_evaluation_outputs(
        self,
        output_dir: Path,
        prefix: str,
        metrics: dict[str, float],
        report: dict,
        matrix: np.ndarray,
        class_names: list[str],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
        write_flat_metrics_csv(output_dir / f"{prefix}_metrics.csv", metrics)
        (output_dir / f"{prefix}_classification_report.json").write_text(json.dumps(report, indent=2))
        write_classification_report_csv(output_dir / f"{prefix}_classification_report.csv", report)
        write_confusion_matrix_csv(output_dir / f"{prefix}_confusion_matrix.csv", matrix, class_names)
        plot_confusion_matrix(output_dir / f"{prefix}_confusion_matrix.png", matrix, class_names)
        write_paper_results_table(
            output_dir / f"{prefix}_paper_table",
            metrics,
            class_names,
            report,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utility classes / functions (unchanged except MetricTracker)
# ─────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, **items: torch.Tensor) -> None:
        self.count += 1
        for key, value in items.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value.detach().cpu())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}


def write_flat_metrics_csv(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def write_classification_report_csv(path: Path, report: dict) -> None:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"label": label, "value": values})
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def plot_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plot_confusion_matrix_with_pillow(path, matrix, class_names)
        return

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    n = len(class_names)
    fig_size = max(5, 0.9 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    cap_names = [c.capitalize() for c in class_names]
    ax.set_xticklabels(cap_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cap_names, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    thresh = 0.5
    for r in range(n):
        for c in range(n):
            val = norm[r, c]
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7 if n > 6 else 9)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_with_pillow(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    cell, label_space = 84, 150
    width  = label_space + cell * len(class_names) + 20
    height = label_space + cell * len(class_names) + 20
    image = Image.new("RGB", (width, height), "white")
    draw  = ImageDraw.Draw(image)
    font  = ImageFont.load_default()
    for index, name in enumerate(class_names):
        x = label_space + index * cell + cell // 2
        draw.text((x - 18, 18), name[:14], fill="black", font=font)
        y = label_space + index * cell + cell // 2
        draw.text((12, y - 5), name[:18], fill="black", font=font)
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            value = float(normalized[row, col])
            shade = int(255 - value * 190)
            x0 = label_space + col * cell
            y0 = label_space + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(shade, shade, 255), outline=(80, 80, 120))
            draw.text((x0 + 18, y0 + 34), f"{value:.3f}", fill="black", font=font)
    draw.text((label_space + cell * len(class_names) // 2 - 24, height - 18), "Predicted", fill="black", font=font)
    draw.text((12, label_space - 24), "True", fill="black", font=font)
    image.save(path)


def write_paper_results_table(
    path: Path,
    metrics: dict[str, float],
    class_names: list[str],
    report: dict,
) -> None:
    accuracy_pct = round(metrics["accuracy"] * 100, 2)
    precision    = round(metrics["precision_macro"], 3)
    recall       = round(metrics["recall_macro"], 3)
    f1           = round(metrics["f1_macro"], 3)
    rows = [
        {"Method": "DICE-FER (ours)", "Setting": "image-based",
         "Expression": str(len(class_names)), "Accuracy (%)": accuracy_pct,
         "Precision": precision, "Recall": recall, "F1 Score": f1},
    ]
    with path.with_suffix(".csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 1.6))
        ax.axis("off")
        col_labels = ["Method", "Setting", "Expressions", "Accuracy (%)", "Precision", "Recall", "F1 Score"]
        cell_data  = [["DICE-FER (ours)", "image-based", str(len(class_names)),
                       f"{accuracy_pct:.2f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]]
        tbl = ax.table(cellText=cell_data, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#EEEDFE")
                cell.set_text_props(weight="bold")
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
