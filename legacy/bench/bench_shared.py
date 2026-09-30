"""Full pipeline with one action actor riding on each inpaint card. Same arguments as `python -m egowhale.batch`.

    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python legacy/bench/bench_shared.py \
        --segment 1 --segment-actors 3 --action 0 --inpaint 4 --depth 3 --out outputs/x <dataset dirs...>

`--action` is ignored: there is one action actor per inpaint card.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from egowhale import batch

ACTION_SHARE = 0.1

_serve = batch.serve


def _shared(jobs, pools, share, inflight, log, board):
    pools["action"] = pools["action_read"] = pools["action_write"] = pools["inpaint"]
    # Inpaint is placed before segment, so the four 0.9 slots land on four cards and segment gets the fifth.
    share = {"inpaint": 1 - ACTION_SHARE, "segment": share["segment"], "action": ACTION_SHARE}
    _serve(jobs, pools, share, inflight, log, board)


batch.serve = _shared

if __name__ == "__main__":
    batch.main()
