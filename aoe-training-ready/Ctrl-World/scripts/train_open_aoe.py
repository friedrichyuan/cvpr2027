from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

_STOP_REQUESTED = False


def _request_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    cli = parser.parse_args(argv)

    ctrl_world_root = Path(cli.ctrl_world_root).resolve()
    sys.path.insert(0, str(ctrl_world_root))

    from config import wm_args
    from dataset.dataset_droid_exp33 import Dataset_mix
    from models.ctrl_world import CrtlWorld

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    args = wm_args()
    apply_overrides(args, cli)

    os.makedirs(args.output_dir, exist_ok=True)
    if cli.wandb_mode:
        os.environ["WANDB_MODE"] = cli.wandb_mode

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_dir=args.output_dir,
    )

    model = CrtlWorld(args)
    if args.ckpt_path:
        print(f"[ctrl_world] loading checkpoint: {args.ckpt_path}", flush=True)
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        load_compatible_state_dict(model, state_dict)
    model.to(accelerator.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataset_cls = OpenAoECtrlWorldDataset if cli.dataset_format == "open_aoe_ctrlworld" else Dataset_mix
    train_dataset = dataset_cls(args, mode="train")
    if accelerator.is_main_process:
        save_training_sample_preview(train_dataset, Path(args.output_dir) / "visualizations")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        args.max_train_steps
        * args.gradient_accumulation_steps
        * total_batch_size
        / max(1, len(train_dataloader))
    )
    if accelerator.is_main_process:
        print("[ctrl_world] training start", flush=True)
        print(f"[ctrl_world] dataset_examples={len(train_dataset)}", flush=True)
        print(f"[ctrl_world] max_train_steps={args.max_train_steps}", flush=True)
        print(f"[ctrl_world] checkpointing_steps={args.checkpointing_steps}", flush=True)
        print(f"[ctrl_world] output_dir={args.output_dir}", flush=True)

    global_step = 0
    loss_accum = 0.0
    loss_count = 0
    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress.set_description("ctrl-world")

    for _epoch in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, _ = model(batch)
                avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean()
                loss_accum += float(avg_loss.item())
                loss_count += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % args.log_steps == 0 and accelerator.is_main_process:
                    train_loss = loss_accum / max(1, loss_count)
                    print(f"step={global_step} train_loss: {train_loss:.8f}", flush=True)
                    progress.set_postfix({"train_loss": train_loss})
                    loss_accum = 0.0
                    loss_count = 0
                if (
                    args.checkpointing_steps
                    and args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                    and accelerator.is_main_process
                ):
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}.pt"
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    print(f"[ctrl_world] saved checkpoint: {save_path}", flush=True)
                if _STOP_REQUESTED:
                    if accelerator.is_main_process:
                        print(f"[ctrl_world] stop requested at step={global_step}", flush=True)
                    save_final_checkpoint(accelerator, model, args.output_dir, global_step)
                    save_final_rgb_rollout(accelerator, model, train_dataset, args, global_step)
                    progress.close()
                    return 0
                if global_step >= args.max_train_steps:
                    save_final_checkpoint(accelerator, model, args.output_dir, global_step)
                    save_final_rgb_rollout(accelerator, model, train_dataset, args, global_step)
                    progress.close()
                    if accelerator.is_main_process:
                        print("[ctrl_world] training end", flush=True)
                    return 0

    progress.close()
    if accelerator.is_main_process:
        save_final_checkpoint(accelerator, model, args.output_dir, global_step)
        save_final_rgb_rollout(accelerator, model, train_dataset, args, global_step)
        print("[ctrl_world] training end", flush=True)
    return 0


def load_compatible_state_dict(model, raw_state_dict: dict) -> None:
    state_dict = raw_state_dict.get("model", raw_state_dict) if isinstance(raw_state_dict, dict) else raw_state_dict
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        elif key in current:
            skipped.append((key, tuple(value.shape), tuple(current[key].shape)))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"[ctrl_world] loaded compatible tensors={len(compatible)} "
        f"skipped_shape_mismatch={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    for key, old_shape, new_shape in skipped[:12]:
        print(f"[ctrl_world] skipped {key}: ckpt={old_shape} model={new_shape}", flush=True)


def save_final_checkpoint(accelerator: Accelerator, model, output_dir: str, step: int) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    save_path = Path(output_dir) / "final_model.pt"
    torch.save(
        {
            "step": int(step),
            "model": accelerator.unwrap_model(model).state_dict(),
        },
        save_path,
    )
    print(f"[ctrl_world] saved final checkpoint: {save_path}", flush=True)


class OpenAoECtrlWorldDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode: str = "train") -> None:
        self.args = args
        self.mode = mode
        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []
        self.prob = args.prob

        dataset_names = args.dataset_names.split("+")
        dataset_cfgs = args.dataset_cfgs.split("+")
        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = Path(args.dataset_meta_info_path) / dataset_cfg / f"{mode}_sample.json"
            with data_json_path.open("r", encoding="utf-8") as f:
                samples = json.load(f)
            dataset_dir = Path(args.dataset_root_path) / dataset_name
            with (Path(args.dataset_meta_info_path) / dataset_name / "stat.json").open("r", encoding="utf-8") as f:
                data_stat = json.load(f)
            state_p01 = torch.tensor(data_stat["state_01"], dtype=torch.float32).numpy()[None, :]
            state_p99 = torch.tensor(data_stat["state_99"], dtype=torch.float32).numpy()[None, :]
            self.dataset_path_all.append([str(dataset_dir) for _ in samples])
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))
            self.norm_all.append((state_p01, state_p99))
            print(f"ALL Open-AoE CtrlWorld dataset, {len(samples)} samples in total", flush=True)
        self.max_id = max(self.samples_len)
        print("samples_len:", self.samples_len, "max_id:", self.max_id, flush=True)

    def __len__(self) -> int:
        return self.max_id

    def _load_latent_video(self, video_path: Path, frame_ids: list[int]) -> torch.Tensor:
        video_tensor = torch.load(video_path, map_location="cpu")
        video_tensor.requires_grad = False
        max_frames = video_tensor.size(0)
        frame_ids = [int(frame_id) if frame_id < max_frames else max_frames - 1 for frame_id in frame_ids]
        return video_tensor[frame_ids]

    @staticmethod
    def normalize_bound(data, data_min, data_max, clip_min: float = -1, clip_max: float = 1, eps: float = 1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        import numpy as np

        return np.clip(ndata, clip_min, clip_max)

    def __getitem__(self, index: int) -> dict:
        import numpy as np

        dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = Path(dataset_path[index])

        frame_now = int(sample["frame_ids"][0])
        ann_file = dataset_dir / self.args.annotation_name / self.mode / f"{sample['episode_id']}.json"
        with ann_file.open("r", encoding="utf-8") as f:
            label = json.load(f)

        frame_len = int(label["video_length"]) - 1
        skip = np.random.randint(1, 3)
        skip_his = int(skip * 4)
        if np.random.random() < 0.15:
            skip_his = 0
        rgb_id = [int(frame_now - i * skip_his) for i in range(self.args.num_history, 0, -1)]
        rgb_id.append(frame_now)
        rgb_id.extend(int(frame_now + i * skip) for i in range(1, self.args.num_frames))
        rgb_id = np.clip(np.asarray(rgb_id), 0, frame_len).astype(int).tolist()
        state_id = np.clip(np.asarray(rgb_id) * int(self.args.down_sample), 0, len(label["action.open_aoe"]) - 1).astype(int)

        latents = []
        for cam_id in range(3):
            rel_path = label["latent_videos"][cam_id]["latent_video_path"]
            latents.append(self._load_latent_video(dataset_dir / rel_path, rgb_id))
        latent = torch.zeros((self.args.num_frames + self.args.num_history, 4, 72, 40), dtype=torch.float32)
        latent[:, :, 0:24] = latents[0]
        latent[:, :, 24:48] = latents[1]
        latent[:, :, 48:72] = latents[2]

        action = np.asarray(label["action.open_aoe"], dtype=np.float32)[state_id]
        action = self.normalize_bound(action, state_p01, state_p99)
        return {
            "text": label["texts"][0],
            "latent": latent.float(),
            "action": torch.tensor(action, dtype=torch.float32),
        }


def save_training_sample_preview(train_dataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_rgb_preview(train_dataset, output_dir):
        return

    try:
        sample = train_dataset[0]
    except Exception as exc:
        print(f"[ctrl_world][warn] failed to load preview sample: {exc}", flush=True)
        return
    for key, value in flatten_tensors(sample).items():
        tensor = value.detach().cpu()
        if tensor.ndim >= 4:
            try:
                save_tensor_video(
                    tensor,
                    output_dir / f"training_sample_{safe_name(key)}_debug.mp4",
                    normalize=True,
                )
                return
            except Exception as exc:
                print(f"[ctrl_world][warn] failed to save preview video {key}: {exc}", flush=True)
    print("[ctrl_world][warn] no video-like tensor found for preview", flush=True)


def save_raw_rgb_preview(train_dataset, output_dir: Path) -> bool:
    label_path = first_annotation_path(train_dataset)
    if label_path is None:
        return False
    try:
        with open(label_path, "r") as f:
            label = json.load(f)
    except Exception as exc:
        print(f"[ctrl_world][warn] failed to read preview annotation {label_path}: {exc}", flush=True)
        return False

    video_entries = label.get("videos") or []
    video_paths = []
    for entry in video_entries:
        rel_path = entry.get("video_path") if isinstance(entry, dict) else None
        if rel_path:
            video_paths.append(label_path.parents[2] / rel_path)
    if not video_paths:
        return False

    try:
        save_multicam_video(video_paths, output_dir / "training_sample_rgb_multicam.mp4")
        return True
    except Exception as exc:
        print(f"[ctrl_world][warn] failed to save raw RGB preview: {exc}", flush=True)
        return False


def first_annotation_path(train_dataset) -> Path | None:
    try:
        sample = train_dataset.samples_all[0][0]
        dataset_dir = Path(train_dataset.dataset_path_all[0][0])
        annotation_name = train_dataset.args.annotation_name
        mode = train_dataset.mode
        return dataset_dir / annotation_name / mode / f"{sample['episode_id']}.json"
    except Exception as exc:
        print(f"[ctrl_world][warn] failed to locate preview annotation: {exc}", flush=True)
        return None


def save_multicam_video(video_paths: list[Path], path: Path, max_frames: int = 64) -> None:
    import imageio.v2 as imageio
    import numpy as np

    readers = [imageio.get_reader(str(video_path)) for video_path in video_paths]
    try:
        lengths = []
        for reader in readers:
            try:
                lengths.append(reader.count_frames())
            except Exception:
                lengths.append(max_frames)
        frame_count = min(max_frames, *lengths)
        frames = []
        for frame_idx in range(frame_count):
            imgs = [reader.get_data(frame_idx)[..., :3] for reader in readers]
            height = min(img.shape[0] for img in imgs)
            imgs = [img[:height] for img in imgs]
            frames.append(np.concatenate(imgs, axis=1))
        imageio.mimsave(path, frames, fps=8)
    finally:
        for reader in readers:
            reader.close()
    print(f"[ctrl_world] saved raw RGB training sample preview: {path}", flush=True)


def save_final_rgb_rollout(accelerator: Accelerator, model, train_dataset, args, step: int) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    try:
        import einops
        import imageio.v2 as imageio
        import numpy as np
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()
        sample = train_dataset[0]
        device = accelerator.device
        video_gt = sample["latent"].unsqueeze(0).to(device)
        actions = sample["action"].unsqueeze(0).to(device)
        texts = [sample["text"]]
        his_latent_gt = video_gt[:, : args.num_history]
        future_latent_gt = video_gt[:, args.num_history :]
        current_latent = future_latent_gt[:, 0]
        with torch.no_grad(), accelerator.autocast():
            action_hidden = unwrapped.action_encoder(
                actions,
                texts,
                unwrapped.tokenizer,
                unwrapped.text_encoder,
                args.frame_level_cond,
            )
            _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
                unwrapped.pipeline,
                image=current_latent,
                text=action_hidden,
                width=args.width,
                height=int(3 * args.height),
                num_frames=args.num_frames,
                history=his_latent_gt,
                num_inference_steps=args.final_visualization_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=args.frame_level_cond,
                his_cond_zero=args.his_cond_zero,
            )

            pred_views = einops.rearrange(pred_latents, "b f c (m h) w -> (b m) f c h w", m=3)
            gt_views = einops.rearrange(torch.cat([his_latent_gt, future_latent_gt], dim=1), "b f c (m h) w -> (b m) f c h w", m=3)
            pred_rgb = decode_latent_video(unwrapped.pipeline, pred_views, args.decode_chunk_size)
            gt_rgb = decode_latent_video(unwrapped.pipeline, gt_views, args.decode_chunk_size)

        pred_rgb = pred_rgb.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)
        gt_rgb = gt_rgb.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)
        pred_rgb = ((pred_rgb / 2.0 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        gt_rgb = ((gt_rgb / 2.0 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        history = gt_rgb[:, : args.num_history]
        pred_with_history = np.concatenate([history, pred_rgb], axis=1)
        frames = []
        for t in range(pred_with_history.shape[1]):
            gt_row = np.concatenate([gt_rgb[v, min(t, gt_rgb.shape[1] - 1)] for v in range(gt_rgb.shape[0])], axis=1)
            pred_row = np.concatenate([pred_with_history[v, t] for v in range(pred_with_history.shape[0])], axis=1)
            frames.append(np.concatenate([gt_row, pred_row], axis=0))
        out_path = Path(args.output_dir) / "visualizations" / f"final_rollout_rgb_step_{step}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=2)
        print(f"[ctrl_world] saved final RGB rollout: {out_path}", flush=True)
        unwrapped.train()
    except Exception as exc:
        print(f"[ctrl_world][warn] failed to save final RGB rollout: {exc}", flush=True)


def decode_latent_video(pipeline, latents: torch.Tensor, decode_chunk_size: int) -> torch.Tensor:
    decoded = []
    batch, frames = latents.shape[:2]
    flat = latents.flatten(0, 1)
    for start in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[start : start + decode_chunk_size] / pipeline.vae.config.scaling_factor
        decoded.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    return torch.cat(decoded, dim=0).reshape(batch, frames, *decoded[0].shape[1:])


def flatten_tensors(value, prefix: str = "") -> dict[str, torch.Tensor]:
    if torch.is_tensor(value):
        return {prefix or "tensor": value}
    if isinstance(value, dict):
        out: dict[str, torch.Tensor] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_tensors(item, name))
        return out
    if isinstance(value, (list, tuple)):
        out: dict[str, torch.Tensor] = {}
        for idx, item in enumerate(value):
            name = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(flatten_tensors(item, name))
        return out
    return {}


def save_tensor_video(tensor: torch.Tensor, path: Path, normalize: bool = False) -> None:
    import imageio.v2 as imageio
    import numpy as np

    arr = tensor.float().numpy()
    while arr.ndim > 4:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D tensor, got {arr.shape}")
    if arr.shape[0] in {1, 3, 4}:
        arr = np.transpose(arr, (1, 2, 3, 0))
    elif arr.shape[1] in {1, 3, 4}:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = np.nan_to_num(arr)
    if normalize:
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        arr = (arr - arr_min) / max(arr_max - arr_min, 1e-8)
    elif arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    frames = (arr * 255).astype(np.uint8)
    imageio.mimsave(path, list(frames), fps=8)
    print(f"[ctrl_world] saved training sample preview: {path}", flush=True)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AoE Ctrl-World training wrapper")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--ctrl-world-root", default=str(default_root))
    parser.add_argument("--svd-model-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--dataset-root-path", required=True)
    parser.add_argument("--dataset-meta-info-path", required=True)
    parser.add_argument("--dataset-names", default="droid_subset")
    parser.add_argument("--dataset-format", default="ctrl_world_droid")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--down-sample", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="aoe_ctrl_world")
    parser.add_argument("--wandb-project-name", default="aoe_ctrl_world")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--max-train-steps", type=int, default=1000000)
    parser.add_argument("--checkpointing-steps", type=int, default=0)
    parser.add_argument("--validation-steps", type=int, default=1000000000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--final-visualization-steps", type=int, default=20)
    return parser


def apply_overrides(args, cli: argparse.Namespace) -> None:
    args.svd_model_path = cli.svd_model_path
    args.clip_model_path = cli.clip_model_path
    args.ckpt_path = None if cli.ckpt_path.lower() in {"", "none", "null"} else cli.ckpt_path
    args.dataset_root_path = cli.dataset_root_path
    args.dataset_meta_info_path = cli.dataset_meta_info_path
    args.dataset_names = cli.dataset_names
    args.dataset_cfgs = cli.dataset_names
    args.action_dim = cli.action_dim
    args.down_sample = cli.down_sample
    args.output_dir = cli.output_dir
    args.tag = cli.tag
    args.wandb_project_name = cli.wandb_project_name
    args.max_train_steps = cli.max_train_steps
    args.checkpointing_steps = cli.checkpointing_steps
    args.validation_steps = cli.validation_steps
    args.log_steps = max(1, cli.log_steps)
    args.train_batch_size = cli.train_batch_size
    args.num_workers = cli.num_workers
    args.learning_rate = cli.learning_rate
    args.gradient_accumulation_steps = cli.gradient_accumulation_steps
    args.mixed_precision = cli.mixed_precision
    args.max_grad_norm = cli.max_grad_norm
    args.final_visualization_steps = cli.final_visualization_steps
    args.video_num = 0


if __name__ == "__main__":
    raise SystemExit(main())
