#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming clip dataset for GenieRedux on large AoE (Open-AoE100_v2 ~100h won't fit in RAM).

The in-RAM `load_clips`/`load_clips_actions` load the whole set as a float tensor (~900GB @100h).
This reads ONE clip per __getitem__ from the per-episode npz (image+action), so memory is O(batch),
independent of dataset size. Build a cached clip index by reading only the small `action` array's
length per file (cheap — does NOT decompress the images). Use with DDP via DistributedSampler.
"""
import os, glob, json, hashlib, random
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def _list_files(npz_dir, skip, max_files):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(npz_dir), "*.npz")))
    assert files, f"no npz in {npz_dir}"
    files = files[skip:]
    if max_files is not None:
        files = files[:max_files]
    return files


class StreamingClipDataset(Dataset):
    def __init__(self, npz_dir, clip_len=16, skip=0, max_files=None, with_action=False, cache_files=4):
        self.clip_len = clip_len
        self.with_action = with_action
        self.files = _list_files(npz_dir, skip, max_files)
        self.index = self._build_index(npz_dir, skip, max_files)
        self.cache_files = cache_files
        self._cache = None                                              # per-worker LRU of decompressed episode arrays

    def _build_index(self, npz_dir, skip, max_files):
        files = self.files
        key = hashlib.md5(
            f"{len(files)}|{skip}|{max_files}|{self.clip_len}|{files[0]}|{files[-1]}".encode()
        ).hexdigest()[:10]
        cache = os.path.join(os.path.expanduser(npz_dir), f"_clipidx_{key}.json")
        if os.path.exists(cache):
            try:
                with open(cache) as fh:
                    return json.load(fh)
            except Exception:
                pass
        index = []
        for fi, f in enumerate(files):
            try:
                n = int(np.load(f)["action"].shape[0]) // self.clip_len  # reads only the small action array
            except Exception:
                n = 0
            for j in range(n):
                index.append((fi, j))
        tmp = cache + f".tmp{os.getpid()}"                                # atomic write (DDP-safe)
        try:
            with open(tmp, "w") as fh:
                json.dump(index, fh)
            os.replace(tmp, cache)
        except Exception:
            pass
        return index

    def __len__(self):
        return len(self.index)

    def _arrays(self, fi):                                              # LRU-cached decompressed episode (avoids re-decompress on repeat hits)
        if self._cache is None:
            self._cache = OrderedDict()
        c = self._cache
        if fi in c:
            c.move_to_end(fi)
            return c[fi]
        d = np.load(self.files[fi])
        item = (d["image"], d["action"]) if self.with_action else (d["image"], None)
        c[fi] = item
        if len(c) > self.cache_files:
            c.popitem(last=False)
        return item

    def __getitem__(self, idx):
        fi, j = self.index[idx]
        img_all, act_all = self._arrays(fi)
        s = j * self.clip_len
        e = s + self.clip_len
        img = img_all[s:e].astype(np.float32) / 255.0                  # (T,H,W,3)
        x = torch.from_numpy(img).permute(3, 0, 1, 2).contiguous()     # (C,T,H,W)
        if self.with_action:
            a = torch.from_numpy(act_all[s:e].astype(np.float32)).contiguous()  # (T,26)
            return x, a
        return x


def count_clips(npz_dir, clip_len=16, skip=0, max_files=None, sample=256):
    """Estimate total clip count (for steps_per_epoch) by sampling files — avoids reading ALL npz at
    startup (which, called on every DDP rank, is 5x-redundant I/O that idles the GPUs for many minutes)."""
    files = _list_files(npz_dir, skip, max_files)
    idx = list(range(len(files)))
    if len(files) > sample:
        random.Random(0).shuffle(idx)
        idx = idx[:sample]
    tot = n = 0
    for i in idx:
        try:
            tot += int(np.load(files[i])["action"].shape[0]) // clip_len
            n += 1
        except Exception:
            pass
    return int((tot / n) * len(files)) if n else 0


class StreamingClipIterable(IterableDataset):
    """Efficient streaming for large/variable episodes: each (rank,worker) owns a disjoint file shard,
    loads each file ONCE and yields ALL its clips (amortizes the npz decompression over all its clips,
    instead of re-decompressing the whole episode per clip). Infinite + self-reshuffling, so the trainer
    controls epoch length via a fixed steps_per_epoch -> DDP-safe (all ranks do the same #steps)."""
    def __init__(self, npz_dir, clip_len=16, skip=0, max_files=None, with_action=False,
                 world=1, rank=0, seed=0):
        self.files = _list_files(npz_dir, skip, max_files)
        self.clip_len = clip_len
        self.with_action = with_action
        self.world = world
        self.rank = rank
        self.seed = seed

    def __iter__(self):
        wi = get_worker_info()
        nw = wi.num_workers if wi else 1
        wid = wi.id if wi else 0
        gid = self.rank * nw + wid                                     # this (rank,worker)'s global id
        gnum = self.world * nw
        mine = self.files[gid::gnum]
        cl = self.clip_len
        ep = 0
        while True:                                                    # cycle (trainer caps via steps_per_epoch)
            rng = random.Random(self.seed + ep * 100003 + gid)
            order = mine[:]
            rng.shuffle(order)
            for f in order:
                try:
                    d = np.load(f)
                    img = d["image"]
                    act = d["action"] if self.with_action else None
                except Exception:
                    continue
                T = img.shape[0] // cl
                js = list(range(T))
                rng.shuffle(js)
                for j in js:
                    s = j * cl
                    e = s + cl
                    x = torch.from_numpy(img[s:e].astype(np.float32) / 255.0).permute(3, 0, 1, 2).contiguous()
                    if self.with_action:
                        yield x, torch.from_numpy(act[s:e].astype(np.float32)).contiguous()
                    else:
                        yield x
            ep += 1
