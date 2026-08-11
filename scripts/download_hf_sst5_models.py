#!/usr/bin/env python3
"""Download SST-5, BERT-Base, and GPT2-Large and verify local inference.

The script deliberately downloads only Transformers configuration, tokenizer,
and Safetensors files for the two models. Hugging Face progress bars remain
enabled, while stage messages are flushed immediately for non-interactive logs.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BERT_REPO_ID = "google-bert/bert-base-uncased"
GPT2_REPO_ID = "openai-community/gpt2-large"
SST5_REPO_ID = "SetFit/sst5"
MODEL_ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download SST-5, BERT-Base, and GPT2-Large from Hugging Face, "
            "then validate the local artifacts without network access."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints",
        help="Model output root (default: %(default)s).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "huggingface",
        help="Dataset output root (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "huggingface" / ".cache",
        help="Temporary Hugging Face cache root (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent Hugging Face model-file downloads (default: %(default)s).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload remote files and replace the saved SST-5 dataset.",
    )
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    return args


def enable_progress_bars() -> None:
    # These must be explicit because cluster environments sometimes disable
    # progress globally. Standard HTTP is used instead of Xet so large model
    # files expose byte-level tqdm progress and resumable partial downloads.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    import datasets
    from huggingface_hub.utils import enable_progress_bars as enable_hub_progress
    from transformers.utils.logging import enable_progress_bar as enable_tf_progress

    datasets.enable_progress_bars()
    enable_hub_progress()
    enable_tf_progress()


def download_model(
    repo_id: str,
    destination: Path,
    cache_dir: Path,
    force_download: bool,
    max_workers: int,
) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    log(f"\n[下载模型] {repo_id}")
    log(f"  保存目录: {destination}")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        cache_dir=str(cache_dir),
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
        force_download=force_download,
        max_workers=max_workers,
    )
    log(f"  下载完成: {snapshot_path}")
    return destination


def _save_dataset_atomically(dataset: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    temporary_dataset = temporary_root / "dataset"
    try:
        dataset.save_to_disk(str(temporary_dataset))
        if destination.exists():
            shutil.rmtree(destination)
        temporary_dataset.replace(destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def download_sst5(
    destination: Path,
    cache_dir: Path,
    force_download: bool,
) -> Path:
    from datasets import DownloadMode, load_dataset, load_from_disk

    log(f"\n[下载数据集] {SST5_REPO_ID}")
    log(f"  保存目录: {destination}")
    if destination.exists() and not force_download:
        try:
            load_from_disk(str(destination))
            log("  已存在有效的本地 SST-5，跳过重复下载。")
            return destination
        except Exception:
            log("  已有目录无法加载，将重新下载并替换。")

    download_mode = (
        DownloadMode.FORCE_REDOWNLOAD
        if force_download
        else DownloadMode.REUSE_DATASET_IF_EXISTS
    )
    dataset = load_dataset(
        SST5_REPO_ID,
        cache_dir=str(cache_dir / "datasets"),
        download_mode=download_mode,
    )
    _save_dataset_atomically(dataset, destination)
    log("  SST-5 下载和本地序列化完成。")
    return destination


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def release_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_sst5(dataset_path: Path) -> str:
    from datasets import load_from_disk

    log("\n[验证数据集] 使用 load_from_disk 离线加载 SST-5")
    dataset = load_from_disk(str(dataset_path))
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(dataset.keys()):
        raise RuntimeError(
            f"SST-5 splits are incomplete: expected {required_splits}, "
            f"found {set(dataset.keys())}."
        )
    for split in sorted(required_splits):
        if not {"text", "label"}.issubset(dataset[split].column_names):
            raise RuntimeError(f"SST-5 {split} is missing text/label columns.")
        if len(dataset[split]) == 0:
            raise RuntimeError(f"SST-5 {split} is empty.")

    example = dataset["validation"][0]
    if not isinstance(example["text"], str) or int(example["label"]) not in range(5):
        raise RuntimeError("SST-5 validation example has an invalid schema or label.")
    sizes = ", ".join(f"{name}={len(dataset[name])}" for name in dataset.keys())
    log(f"  验证通过: {sizes}")
    return example["text"]


def validate_bert(model_path: Path, sentence: str, device: str) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    log(f"\n[验证模型] BERT-Base，device={device}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(model_path), local_files_only=True
    ).to(device)
    model.eval()
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    shape = tuple(outputs.last_hidden_state.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[2] != model.config.hidden_size:
        raise RuntimeError(f"Unexpected BERT output shape: {shape}.")
    log(f"  前向通过: last_hidden_state.shape={shape}")
    release_model(model)


def validate_gpt2(model_path: Path, sentence: str, device: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"\n[验证模型] GPT2-Large，device={device}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True
    ).to(device)
    model.eval()
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    logits_shape = tuple(outputs.logits.shape)
    if (
        len(logits_shape) != 3
        or logits_shape[0] != 1
        or logits_shape[2] != model.config.vocab_size
    ):
        raise RuntimeError(f"Unexpected GPT2 output shape: {logits_shape}.")
    generated_text = tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)
    if not generated_text:
        raise RuntimeError("GPT2 generated an empty string.")
    log(f"  前向通过: logits.shape={logits_shape}")
    log(f"  生成通过: {generated_text!r}")
    release_model(model)


def main() -> None:
    args = parse_args()
    enable_progress_bars()

    models_dir = args.models_dir.expanduser().resolve()
    datasets_dir = args.datasets_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    bert_path = models_dir / "bert-base-uncased"
    gpt2_path = models_dir / "gpt2-large"
    sst5_path = datasets_dir / "sst5"

    log("Hugging Face 下载与验证任务")
    log(f"  模型根目录: {models_dir}")
    log(f"  数据根目录: {datasets_dir}")
    log(f"  缓存根目录: {cache_dir}")

    download_sst5(sst5_path, cache_dir, args.force_download)
    download_model(
        BERT_REPO_ID,
        bert_path,
        cache_dir / "hub",
        args.force_download,
        args.max_workers,
    )
    download_model(
        GPT2_REPO_ID,
        gpt2_path,
        cache_dir / "hub",
        args.force_download,
        args.max_workers,
    )

    device = resolve_device(args.device)
    sentence = validate_sst5(sst5_path)
    validate_bert(bert_path, sentence, device)
    validate_gpt2(gpt2_path, sentence, device)

    log("\n全部下载完成，SST-5、BERT-Base 和 GPT2-Large 均验证通过。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n任务已由用户中断。已下载文件会保留，可在下次运行时续用。")
        raise SystemExit(130)
    except Exception as error:
        log(f"\n任务失败: {type(error).__name__}: {error}")
        raise
