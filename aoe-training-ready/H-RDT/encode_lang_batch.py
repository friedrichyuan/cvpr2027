#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Encode language descriptions from AoE HDF5 files using T5-v1.1-XXL.

Reads 'llm_description' from each HDF5 file and saves T5 embeddings
as .pt files alongside the HDF5 files.

Output per file: <index>.pt containing:
  {"instruction": str, "embeddings": tensor(seq_len, 4096), ...}
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from multiprocessing import Process, Queue

import torch
import yaml
import h5py
from tqdm import tqdm

PROJECT_ROOT = os.environ.get(
    'HRDT_PROJECT_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
sys.path.append(PROJECT_ROOT)
from models.encoder.t5_encoder import T5Embedder


def collect_files(data_root):
    """Collect all HDF5 files that need language embedding."""
    files = []
    for episode in sorted(os.listdir(data_root)):
        episode_dir = os.path.join(data_root, episode)
        if not os.path.isdir(episode_dir):
            continue
        for fname in sorted(os.listdir(episode_dir)):
            if not fname.endswith('.hdf5'):
                continue
            hdf5_path = os.path.join(episode_dir, fname)
            idx = fname.replace('.hdf5', '')
            pt_path = os.path.join(episode_dir, f"{idx}.pt")
            if not os.path.exists(pt_path):
                files.append({
                    'hdf5_path': hdf5_path,
                    'pt_path': pt_path,
                    'episode': episode,
                    'index': idx,
                })
    return files


def worker(process_id, gpu_id, file_list, progress_queue, t5_model_path, config_path):
    """Worker process: load T5 model and encode descriptions."""
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        text_embedder = T5Embedder(
            from_pretrained=t5_model_path,
            model_max_length=config['dataset']['tokenizer_max_length'],
            device=device,
        )
        tokenizer = text_embedder.tokenizer
        text_encoder = text_embedder.model

        processed = 0
        failed = 0

        for fi in file_list:
            try:
                with h5py.File(fi['hdf5_path'], 'r') as f:
                    desc = f.attrs.get('llm_description', '')
                    if isinstance(desc, bytes):
                        desc = desc.decode('utf-8')

                if not desc:
                    failed += 1
                    progress_queue.put(('failed', process_id))
                    continue

                tok = tokenizer([desc], return_tensors='pt', padding='longest', truncation=True)
                tokens = tok['input_ids'].to(device)
                attn = tok['attention_mask'].to(device)

                with torch.no_grad():
                    embeds = text_encoder(
                        input_ids=tokens,
                        attention_mask=attn,
                    )['last_hidden_state'].detach().cpu()

                attn_cpu = attn.cpu().bool()
                text_embed = embeds[0][attn_cpu[0]]

                torch.save({
                    'instruction': desc,
                    'embeddings': text_embed,
                    'episode': fi['episode'],
                    'file_index': fi['index'],
                }, fi['pt_path'])

                processed += 1
                progress_queue.put(('processed', process_id))

            except Exception as e:
                print(f"Process {process_id}: error on {fi['hdf5_path']}: {e}")
                failed += 1
                progress_queue.put(('failed', process_id))

        progress_queue.put(('done', process_id, processed, failed))

    except Exception as e:
        import traceback
        print(f"Process {process_id} fatal error:\n{traceback.format_exc()}")
        progress_queue.put(('error', process_id, str(e)))


def main():
    parser = argparse.ArgumentParser(description='Encode AoE language embeddings with T5')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root of processed data (contains train/test)')
    parser.add_argument('--t5_model_path', type=str,
                        default=os.environ.get('T5_MODEL_PATH', '/data/lingxuan/weights/t5-v1_1-xxl'),
                        help='Path to T5-v1.1-XXL weights')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'configs/hrdt_aoe_pretrain.yaml'),
                        help='YAML config path')
    parser.add_argument('--num_gpus', type=int,
                        default=int(os.environ.get('NUM_GPUS', 1)))
    parser.add_argument('--processes_per_gpu', type=int,
                        default=int(os.environ.get('PROCESSES_PER_GPU', 1)))
    args = parser.parse_args()

    total_processes = args.num_gpus * args.processes_per_gpu

    print("Collecting files...")
    all_files = collect_files(args.data_root)
    if not all_files:
        print("No files to process!")
        return

    print(f"Found {len(all_files)} files to encode")

    # Distribute files
    file_lists = [[] for _ in range(total_processes)]
    for i, fi in enumerate(all_files):
        file_lists[i % total_processes].append(fi)

    progress_queue = Queue()

    # Monitor
    monitor = Process(
        target=_progress_monitor,
        args=(len(all_files), progress_queue, total_processes),
    )
    monitor.start()

    # Workers
    processes = []
    for i in range(total_processes):
        gpu_id = i // args.processes_per_gpu
        p = Process(
            target=worker,
            args=(i, gpu_id, file_lists[i], progress_queue,
                  args.t5_model_path, args.config_path),
        )
        p.start()
        processes.append(p)
        time.sleep(0.3)

    for p in processes:
        p.join()
    monitor.join()

    print("\nDone!")


def _progress_monitor(total, progress_queue, num_processes):
    pbar = tqdm(total=total, desc="Encoding language")
    finished = 0
    while finished < num_processes:
        try:
            msg = progress_queue.get(timeout=1)
            if msg[0] in ('processed', 'failed'):
                pbar.update(1)
            elif msg[0] == 'done':
                finished += 1
            elif msg[0] == 'error':
                finished += 1
        except Exception:
            continue
    pbar.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
