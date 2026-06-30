# Open-AoE Action Specification (hand-as-EEF + camera)

This document defines the state/action representation produced by the Open-AoE
downstream-model converters in this directory and consumed by the world-model / VLA
recipes. It describes **what each recipe actually trains on**. The principle is
*correct and clearly documented* — wrist 6-DoF motion and camera motion are real;
fingers are an open/close proxy (no MANO forward kinematics yet).

## State (22D, camera frame)

Per frame, a dual-hand "hand-as-end-effector" state:

```text
[0:3]    left  wrist xyz
[3:9]    left  wrist rotation (6D = first two columns of the rotation matrix)
[9]      left  gripper proxy (open/close scalar in [0, 1])
[10:13]  right wrist xyz
[13:19]  right wrist rotation (6D)
[19]     right gripper proxy
[20]     left  hand valid (0/1)
[21]     right hand valid (0/1)
```

## Action

- **Hand 20D** — the motion part of the state (`[0:20]`) as next-frame absolute or
  per-frame delta in the camera frame. Wrist 6-DoF motion is real; fingers are an
  open/close proxy (MANO fingertip positions are not reconstructed).
- **Camera 6D** (optional, appended → **26D**) — relative camera self-motion between
  adjacent frames, `inv(cam_c2w[t]) @ cam_c2w[t+1]` → translation(3) + axis-angle(3).

Key finding across the Open-AoE world-model recipes: first-person dynamics are
**dominated by camera ego-motion**. The camera 6D is therefore treated as a
first-class action component (e.g. iVideoGPT 26D), or a small amount of hand
supervision is used to ground unsupervised latent actions to the hand (laom).

## How each recipe uses it

| Recipe | Category | Action usage | Dim |
|--------|----------|--------------|-----|
| [`ivideogpt/`](ivideogpt/) | World model (action-conditioned) | action as conditioning input | 26D (hand + camera) |
| [`smolvla/`](smolvla/) | VLA | action as policy output | 20D (hand) / 26D (hand + camera) |
| [`laom/`](laom/) | Latent-action world model | unsupervised latent + 20D hand as supervision label | 20D label |
| [`genie-redux/`](genie-redux/) | Latent-action world model | unsupervised (video only); guided variant conditions on true 26D | — / 26D |
| [`adaworld/`](adaworld/) | Latent-action model | unsupervised frame pairs | — |
| [`dreamdojo/`](dreamdojo/) | World model | LAM-inferred latent action (zero-shot) | — |

## MANO note

The current representation is wrist 6-DoF + finger open/close proxy (no MANO FK).
An optional MANO forward-kinematics upgrade (21 joints + 5 fingertips, e.g. toward a
wrist + fingertip action) does not change the recipes' core behaviour, which depends
on the wrist 6-DoF + camera motion rather than fingertip precision. The MANO model is
license-gated and never committed — see the root [`LEGAL.md`](../LEGAL.md).

> Note: this **26D = hand 20D + camera 6D** is the egocentric world-model action and
> is distinct from the retargeting side's 28D robot joint action; they are different
> vectors for different purposes.
