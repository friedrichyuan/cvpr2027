

<font style="background-color:#FBDE28;">DDL：</font>

+ <font style="background-color:#FBDE28;">7月中下旬 WAIC （Open-AoE）</font>
+ <font style="background-color:#FBDE28;">9月中 外滩大会 </font>

## 背景及价值：


1. 影响力价值：开源数据集对公司及团队在行业影响力上的价值
2. 商业化价值：开源数据使得我们的数据更容易被潜在客户公司的算法团队看到并试用，带来更大规模闭源数据需求，有利于扩大AoE数据商业化规模
3. 技术力价值：AoE技术团队获取社区密集反馈和贡献，加速技术迭代保持竞争力

## 开源原则：
1. **数据采集平台/加工标注质检能力**为AoE数据生产的核心竞争力及商业壁垒，不会开源（类比Meta开源Ego4D眼镜数据集，但不会开源Aria采集设备和算法）
2. **开源数据**需要代表AoE数据的最高水准，多轮人工复审，社区口碑崩盘的影响力规模远大于单点客户POC失败
3. **开源AoE工具链**是为了让AoE数据更易用、更好用，扩大在具身公司/研究机构间的影响力，工具链需要动态链接AoE原始数据到行业最热门模型路线（比如pi-0.5/gr00t/fastWAM/以及后续持续跟进打通爆款开源模型）




## 开源计划：
<!-- 这是一张图片，ocr 内容为：OPEN-AOE:打通原始数据到模型训练的最后一公里 1.开源.OPEN SOURCE(开源交付物矩阵) 已有实现 已有实现 AOE-TRAINING-READY 工具链 AOE-DISPLAY OPEN-AOE-1000H OPEN-AOE TECH REPORT AOE-RETARGET-REPLAY 工具链 形态:1000H 视频+ATOMIC-ACTION 形态:RETARGET_NPZ_TO_LEROBOT.PY 形态:ANT_AOE_RETARGET_TO_G1.PY+ 形态:4 页短报告/AOE 1.0V3 章节 形态:基于RERUN 的数据浏览器 双语描述+MANO标注+相机轨迹 PIO.5/GR00T/FASTWAM 等模型 RECIPE TRIPLE_PLAY.PY 的产品化封装 (类似 ROPEDIA) 价值锚点:开源数据分布洞察+ 深度(按片段) 价值锚点:对有真机的团队,直接 价值锚点:打通AOE数据到行业 价值锚点:方便使用者进行数据 轻量数据质量实验 价值锚点:1000H  是被加入基模 最热门模型路线的最后一公里 洞察,过滤,对比 REPLAY 到真机,直观体现 AOE 的 PRETRAIN 的入门门槛,让使用者有 当前完成度:0.预期 1-2周 动作学习价值 数据可用,形成路径依赖 当前完成度:0.MVP 2-3周 当前完成度:60%,LEROBOT V2.1+ 当前完成度:80%,核心脚本已可用 当前完成度:数据/标注流水线已存在, FASTWAM 已验证 需筛选打包 2.OPEN-AOE TOOLCHAIN PIPELINE(工具链流水线) [原示] 可洞察 AOE-DISPLAY (RERUN) TECH REPORT 脚本 沿用 ANT_AOE_RETARGET_TO_G1 原始AOE 数据: 可对外发布 [洞察] (分布分析+质量实验) 一外部贡献回流 TRIPLE_PLAY MP4 UNDISTORTED + TRIPLE_PLAYPY HANDS.NPZ + (EGO+MANO+MUJOCO 2X2 可视化)  ANT_AOE_RETARGET_TO_G1.PY [真机] G1.NPZ 可上真机 CAMERA TRAJ.NPZ (重定向与映射) 真机回放SDK DK(按机型适配) 双语标注 已通过 FASTWAM (已验证) FASTWAM MULTIHEAD(E STEP 100 LOSS 0.523 LEROBOT V2.1 RETARGET_NPZ.TO LEROBOT.PY [训练] 验证 PIO.5 德调 RECIPE (待打通) (28D JOINT SPACE) (格式转换) 可训模型 GROOT  微调 RECIPE(待打通) 持续跟进煤款开源模型 3.平台开放,PLATFORM ONLY(平台开放层) A.AOE数据采集APP B.AOE清洗加工标注链路 形态:移动端APP安装包/应用商店上架 形态:后台 SAAS/内部工具账号 开放对象:任何用户(含个人开发者/爱好者) 开放对象:定向开放给高校老师(科研合作) 价值锚点:让任何人用自己的手机/眼镜接入AOE采集协议. 价值错点:让高校研究者按自已实验目标组织新的采集+标注任务, 扩大数据源:社区夷献入口 沉淀更多0GO数据资产 当前方式:免费下载使用,不开源源代码 开放方式:申请制账号接入,不开源源代码 4.关键技术决策(KEY  TECHNICAL DECISIONS) 1000H  子朱筛选门限: 开源数据+闭源平台双轨: 28D JOINT SPACE 与FASTWAM/MULTIHEAD 不强推自定义ZARR,直接交付 开源吸引使用者,平台沉淀贡献者 完全一致:[L_ARM(7),R_ARM(7). MP4+JSON+NPZ;额外提供 PRED VALID 覆盖率+IK FAILURE RATE+ LEROBOT V2.1 转换器 L_HAND(6),R_HAND(6),PAD(2)] 相机轨迹连续性 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/65256396/1779878026553-ce9a70d9-f7d6-4aa5-9337-2f64a01cad6f.png)

<!-- 这是一张图片，ocr 内容为：OPEN-AOE:打通原始数据到模型训练的最后一公里 AOE-RETARGET-REPLAY 已有实现 已有实现 已有实现 OPEN-AOE AOE-TRANING-READY 2 OPEN-AOE-10000H AOE-DISPLAY 工具链 TECH REPORT 工具链 自形态:1000H 视频+ATOMIC-ACTION 形态:RETARGET_NPZ_TO_LEROBOT.PY 形态:ANT_AOE_RETARGET_TO_G1.PY 形态:4页短报告/ 形态:基于RERUN的数据 双语描述+MANO标注+相机轨迹 +PIO.5/GR00T / FASTWAM 等 +TRIPLE_PLAY.PY 的产品化封装 浏览器(类似 ROPEDIA) AOE 1.0 V3 章节 +深度(按片段) 模型 RECIPE 价值锚点:对有真机的团队, 价值锚点:1000H 是被加入基模 价值锚点:打通AOE数据到 价值锚点:方便教用者 价值锚点:开源数据分布 直接 REPLAY 到真机,直观体现 行业最热门模型路线的最后一公里 PRETRAIN 的入门门槛,让使用者有 进行数据洞察,过滤,对比 洞察+轻量数据质量实验 AOE 的动作学习价值 数据可用,形成路径依赖 当前完成度:60% 当前完成度:80%, 当前完成度:0, 当前完成度:0, 当前完成度:数据/标注流水线 沿用 已通过FASTWAM LEROBOT V2.1+FASTWAM ANT AOE RETARGET TO G 核心脚本已可用 MVP 2-3周 STEP 100 预期1-2周 已存在,需筛选打包 已验证 TRIPLE_PLAY LOSS 0.523 验证 OPEN-AOE TOOLCHAIN PIPELINE [展示]AOE-DISPLAY (RERUN) TAG:可洞察 [洞察]TECH REPORT 脚本 TAG:可对外发布 原始AOE数据: (分布分析+质量实验) TRIPLE_PLAY.PY (EGO+MANO+MUJOCO 2X2可视化) MP4 UNDISTORTED [真机]ANT_AOE_RETARGET_TO_G1.PY G1.NPZ + HANDS.NPZ 真机回放SDK(按机型适配) TAG:可上真机 +CAMERA_TRAJ.NPZ FASTWAM MULTIHEAD (已验证) +双语标注 PIO.5 微调 RECIPE (待打通) [训练] RETARGET_NPZ_TO_LEROBOT.PY TAG:可训模型 LEROBOT V2.1(28D JOINT SPACE) GROOT 微调 RECIPE(待打通) 持续跟进爆款开源模型 不强推自定义 ZARR,直接交付 E与R 28D JOINT SPACE 1000H 子集筛选门限:PRED_VALID FASTWAM/MULTIHEAD KEY TECHNICAL DECISIONS 覆盖率+IK_FAILURE_RATE+ MP4+JSON+NPZ;额外提供 完全一致:[L_ARM(7),R_ARM(7), (关键技术决策) 相机轨迹连续性 LEROBOT V2.1转换器 L_HAND(6),R_HAND(6),PAD(2)] -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/65256396/1779854712143-ebf11c67-bbb3-4b7a-8447-caf3f9100ccc.png)

| 内容 | 价值 | 责任人/贡献者 |
| --- | --- | --- |
| Open-AoE-2000H<br/>（2000小时带原子动作描述和MANO标注的AoE数据） | （1000小时是被加入基模pretrain的入门门槛）<br/>让使用者有数据可用，对AoE数据形成一定路径依赖 |  |
| Open-AoE tech report<br/>4页短报告，或者更新到AoE 1.0 v3中 | 对开源数据分布的洞察（场景/动作/时间/人员/采集机型等数据分布）<br/>轻量的数据质量实验 |  |
| AoE-display | 类似Ropedia的rerun数据可视化，方便使用者进行数据洞察 |  |
| AoE-retarget-replay工具链 | 对于有真机的团队，可以直接把AoE数据的动作replay到真机上，直观体现AoE数据的动作学习价值 |  |
| AoE-training-ready工具链 | 打通AoE数据到行业最热门模型路线的最后一公里（比如pi-0.5/gr00t/fastWAM/以及后续持续跟进打通爆款开源模型） |  |
|  | | |






## 参考方案：EgoVerse
[https://github.com/GaTech-RL2/EgoVerse](https://github.com/GaTech-RL2/EgoVerse)

EgoVerse 是一个面向**第一人称人类演示数据**的机器人学习框架，目标是把 egocentric 演示 + 多 embodiment 机器人数据，**统一加工成可训练的策略学习数据**，并提供从采集、处理、存储、训练到评估的完整链路。

<!-- 这是一张图片，ocr 内容为：TRAINING & 1 MODELS:HPT(BC+FLOW)/ TRAINHYDRAPY VISUALIZATION & QA /Y EVALUATION LAYER (HYDRA +PYTORCH LIGHTNING DDP) ACT/PIO.5/EGOBRIDGE LOAD + TRANSFORM DATA LOADING 2 MULTIDATASET DATASCHEMATIC+ DATASETFILTER EPISODERESOLVER 管 ZARRDATASET LAYER (S3/LOCAL) (SQL LAMBDA) NORMALIZATION ZARREPISODE LATENT LNSPECTOR (DASH APP) TRANSFORM EMBODIMENT.GET_TRANSFORM_LIST() 3 COMPONENTS:INTERPOLATEPOSE/ COORDINATEFRAMETRANSFORM/CONCATKEYS/ PIPELINE SPLITKEYS/QUATTOYPR/ACTIONCHUNKING CONVENTION: DATA FORMAT REQUIRED ARRAYS: 4 OPTIONAL ARRAYS: ZARRWRITER XYZWXYZ QUATERNION/ IMAGES.FRONT_1. LEFT/RIGHT.OBS_EE_POSE, OBS_KEYPOINTS,ANNOTATIONS, VELOCITY_DISTRIBUTION/ LAYER ZARR V3 SLAM WORLD/UTC <EPISODE_HASH>.ZARR/ OBS_HEAD_POSE DINO.,QWEN.* TIMESTAMP HASH ZARR VALIDATOR 5 DATA INGESTION LANGUAGE: EMBEDDING: ARIA_TO_ZARR/EVA_TO_ZARR/ SCALE_TO_ZARR_ANNOTATION+ DINOV3/QWEN3 LAYER MECKA_TO_ZARR/SFS_TO_EGOVERSE_ZARR LLM CONYERTER PIPELINE ZARREGISTER RAW ANNOTATION/ STORAGE & POSTGRESQL   TABLEROW: CLOUDFLARE R2(S3-COMPATIBLE) EPISODE VIZ EPISODE_HASH/OPERATOR / LAB/ TASK/EMBODIMENT/ REGISTRY LAYER BUCKET RLDB NUM_FRAMES/ZARR_PATH/IS_EVAL/EVAL_SCORE 7 DATA UPLOAD ARIA_UPLOADER / EVA_UPLOADER ABSTRACT_UPLOAD.PY LAYER S3://../RAW_V2/{EMBODIMENT]/ -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/65256396/1779853810561-ea542d24-b380-42a4-a2d9-b90fe18f9add.png)

### 七层架构（自下而上）
1. **Data Upload Layer**：`abstract_upload.py` + `aria_uploader / eva_uploader`，原始数据上传到 S3 `raw_v2/{embodiment}/`。
2. **Storage & Registry Layer**：Cloudflare R2 对象存储 + PostgreSQL 注册表（episode_hash、task、embodiment、num_frames、zarr_path 等字段）。
3. **Data Ingestion Layer**：`aria_to_zarr / eva_to_zarr / mecka_to_zarr / sfs_to_egoverse_zarr` 等转换脚本，外加语言标注转换（Scale + LLM）与 DINOv3 / Qwen3 embedding 预计算。
4. **Data Format Layer (Zarr v3)**：`ZarrWriter` 写出**自包含的 episode 目录**，约定见下。
5. **Transform Pipeline**：由 `Embodiment.get_transform_list()` 编排，含坐标系变换、key 重映射、拼接拆分、四元数→YPR、action chunking。
6. **Data Loading Layer**：`MultiDataset → ZarrDataset → ZarrEpisode`；`EpisodeResolver`（S3/Local）+ `DatasetFilter`（SQL lambda）+ `DataSchematic` + 归一化统计。
7. **Training & Evaluation Layer**：`trainHydra.py` + PyTorch Lightning DDP；模型有 HPT (BC+Flow)、ACT、Pi0.5、EgoBridge。

辅助：**Visualization & QA Layer**（Latent Inspector / velocity 分布 / 标注可视化）横贯多层。

### Zarr v3 Episode 格式约定
每个 episode 是一个 `.zarr/` 目录。

| 类别 | 关键数组 | 形状 |
| --- | --- | --- |
| 必须 | `images.front_1` | (T,) VarLenBytes，JPEG |
| 必须 | `left/right.obs_ee_pose` | (T, 7) XYZWXYZ |
| 必须 | `obs_head_pose` | (T, 7) XYZWXYZ |
| 可选 | `left/right.obs_keypoints` | (T, 63)，21×3 |
| 可选 | `annotations` | (T,) JSON |
| 可选 | `dino.front_1` / `qwen.annotations` | 预计算 embedding |


核心约定：原始数据写入时存 SLAM world 系，训练时由 `ActionChunkCoordinateFrameTransform` **在线**转到 head-relative；四元数顺序为 **XYZWXYZ**（前 3 位平移 + 后 4 位 w,x,y,z）；episode 唯一 ID 为 `YYYY-MM-DD-HH-MM-SS-ffffff`。

### Embodiment 抽象
`Embodiment` ABC 统一各种数据源接入，每个子类只需实现 `_get_keymap(mode)` 与 `get_transform_list(mode)`；当前覆盖 **Aria、Eva、Mecka、Scale** 四类。

### 当前最活跃的开发方向（社区趋势）
+ **HNet / DFoT（Diffusion Forcing of Tokens）**：当前最大研究投入，含 packed training、层级网络、autoregressive staircase sampler。
+ **语言条件训练**：Scale 标注 → LLM 改写 → 写入 `annotations` 数组，Qwen3 embedding 预计算。
+ **跨 embodiment 对齐（EgoBridge）**：OT + DTW 损失对齐人-机表征。
+ **多坐标系（head / wrist）+ 6-DoF / YPR 切换**。
+ **Action Expert Dataset**：BOS/EOS 锚定采样，匹配带分段标注的数据。
+ **Pi0.5 大模型微调**：OpenPi 子模块持续优化。
+ **可视化与 QA 工具**：Latent Inspector、Zarr validator、velocity 分布等密集合入。

代码库现状：**<font style="background-color:#FBDE28;">250+ 远程分支，40+ 贡献者，月级别新增 PR 20+</font>**，正在从“单 embodiment BC”演进为“多模态多 embodiment 统一学习平台”。





## 讨论：
### 0527讨论：
目标：对齐背景价值，开源范畴，大致的分工和节奏

亮点/影响力：

+ 浙大合作、[@长涛](https://www.yuque.com/miaochangtao.mct)
+ 数据可视化网页、
+ 社区持续运营（前沿方法适配）



<!-- 这是一张图片，ocr 内容为：ANT TECH DAY 轨迹解决"怎么动",文本标注解决"为什么动" 527辆效技术日 基于多模态智能体的自动文本标注链路 视频片段+轨迹 人工标注问题 慢 规模上来后人工标注 吞吐跟不上 贵 每小时数据的标注 成本不可忽略 不一致 同一动作可能被标成 不同粒度 TASK(任务) 场景) CATEGORY ACTION(动作) 目标 形成稳定的 SCENE / TASK / ACTION 厨房维修仓储洗车 放置 抓取 拧紧 清洁 切割 分拣 检修 做饭 擦拭 三级标签体系 轨迹提供稳定的"怎么动"信号,文本标注补齐"为什么动"的语义,二者结合,构建可扩展.高一致性的三层标签体系 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/65256396/1779872057842-f84c2ba4-8a58-41c8-9e2d-642248bf0e91.png)

开放文本标注agent的设计作为社区贡献？比赛？

