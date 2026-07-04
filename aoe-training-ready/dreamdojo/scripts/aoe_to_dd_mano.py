#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AoE -> DreamDojo MANODataset (egodex route) builder.
Per AoE recording: HDF5 (wrist + 20 ARKit-node rot6d, camera frame) + mp4 symlink (loader resizes).
Per-node global rotation = wrist_global @ chain(local hand_pose) down standard MANO tree.
rot6d convention matches groot_dreams dataset_mano.py: first two COLUMNS of R.

No MANO .pkl / model file is used or shipped: the finger kinematics are the standard MANO
joint-parent tree (structural constants below), applied analytically to the AoE hand_pose.

Usage (paths via args; nothing hard-coded):
  python aoe_to_dd_mano.py --data-root /PATH_TO/Open-AoE/poc_deliver \
      --out $AOE_EGODEX_HDF5_ROOT --video-root $AOE_EGODEX_VIDEO_ROOT --limit 500
Writes  $out/part0/<seg>.hdf5  and links  $video-root/part0/<seg>.mp4 .
"""
import os, argparse, glob
import numpy as np, h5py

LEFT_IDX, RIGHT_IDX = 0, 1  # AoE pred_*[0]=left,[1]=right (assumption; validate)
MANO_PARENT = [-1,0,1,2, 0,4,5, 0,7,8, 0,10,11, 0,13,14]   # 0=wrist; hand_pose(45)=joints 1..15
MANO_FINGER = {"index":[1,2,3],"middle":[4,5,6],"little":[7,8,9],"ring":[10,11,12],"thumb":[13,14,15]}
ARKIT_FINGERS=["thumb","index","middle","ring","little"]; ARKIT_JOINTS=["Knuckle","IntermediateBase","IntermediateTip","Tip"]
FMAP={"thumb":"Thumb","index":"IndexFinger","middle":"MiddleFinger","ring":"RingFinger","little":"LittleFinger"}
IDENT6=np.array([1,0,0,0,1,0],dtype=np.float32)

def node_names(side): return [f"{side}{FMAP[f]}{j}" for f in ARKIT_FINGERS for j in ARKIT_JOINTS]

def aa_to_mat(aa):
    aa=aa.astype(np.float64); th=np.linalg.norm(aa,axis=-1,keepdims=True); small=th<1e-8
    ax=aa/np.where(small,1.0,th); x,y,z=ax[...,0],ax[...,1],ax[...,2]
    c=np.cos(th)[...,0]; s=np.sin(th)[...,0]; C=1.0-c
    R=np.empty(aa.shape[:-1]+(3,3))
    R[...,0,0]=c+x*x*C; R[...,0,1]=x*y*C-z*s; R[...,0,2]=x*z*C+y*s
    R[...,1,0]=y*x*C+z*s; R[...,1,1]=c+y*y*C; R[...,1,2]=y*z*C-x*s
    R[...,2,0]=z*x*C-y*s; R[...,2,1]=z*y*C+x*s; R[...,2,2]=c+z*z*C
    return np.where(small[...,None],np.eye(3),R)

def mat_to_rot6d(R): return np.concatenate([R[...,:,0],R[...,:,1]],axis=-1).astype(np.float32)  # cols 0,1

def hand(prc_h, php_h):
    T=prc_h.shape[0]; G=[None]*16; G[0]=aa_to_mat(prc_h)
    for j in range(1,16): G[j]=np.matmul(G[MANO_PARENT[j]], aa_to_mat(php_h[:,3*(j-1):3*(j-1)+3]))
    per={}
    for f in ARKIT_FINGERS:
        mj=MANO_FINGER[f]
        for k,jn in enumerate(["Knuckle","IntermediateBase","IntermediateTip"]): per[(f,jn)]=mat_to_rot6d(G[mj[k]])
        per[(f,"Tip")]=np.tile(IDENT6,(T,1))
    return G[0], per

def find_video(sd):
    for pat in ["ego_process/ego_undistorted_video/*.mp4","ego_process/**/*undistort*.mp4","**/*undistort*.mp4","**/*.mp4"]:
        g=sorted(glob.glob(os.path.join(sd,pat),recursive=True))
        if g: return g[0]
    return None

def build(sd, out_root, vid_root):
    stem=os.path.basename(sd.rstrip("/"))
    hn=glob.glob(os.path.join(sd,"**/hands.npz"),recursive=True)
    if not hn: return f"{stem}: no hands.npz"
    mp4=find_video(sd)
    if not mp4: return f"{stem}: no mp4"
    d=np.load(hn[0]); prc,ptc,php=d["pred_rot_cam"],d["pred_trans_cam"],d["pred_hand_pose"]; T=prc.shape[1]
    o=os.path.join(out_root,"part0",stem+".hdf5"); os.makedirs(os.path.dirname(o),exist_ok=True)
    with h5py.File(o,"w") as f:
        for side,idx in [("left",LEFT_IDX),("right",RIGHT_IDX)]:
            wR,nodes=hand(prc[idx],php[idx])
            f.create_dataset(f"wrist/{side}_pose_cam_rot6d",data=np.concatenate([ptc[idx].astype(np.float32),mat_to_rot6d(wR)],axis=1))
            g=f.create_group(f"rot6d/{side}")
            for name,(ff,jj) in zip(node_names(side),[(a,b) for a in ARKIT_FINGERS for b in ARKIT_JOINTS]):
                g.create_dataset(name,data=nodes[(ff,jj)])
    vd=os.path.join(vid_root,"part0",stem+".mp4"); os.makedirs(os.path.dirname(vd),exist_ok=True)
    if os.path.lexists(vd): os.remove(vd)
    os.link(os.path.abspath(mp4),vd)
    return f"{stem}: OK T={T} hdf5={o} mp4<-{os.path.basename(mp4)}"

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--video-root",required=True); ap.add_argument("--limit",type=int,default=1)
    a=ap.parse_args()
    sds=sorted([d for d in glob.glob(os.path.join(a.data_root,"*")) if os.path.isdir(d)])[:a.limit]
    for sd in sds: print(build(sd,a.out,a.video_root))
