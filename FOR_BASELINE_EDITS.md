# RoboTwin 分支相对原库的最终变化

## 对比范围

- 分叉基线：`a2c298b0a3df3778b973fe65e9e58877b292d8a7`
- 最新提交：`bddb98673471a32ea11603c3fbca2a03f80729da`
- Diff 范围：`a2c298b..bddb986`
- 最终净变化：14 个文件，新增 733 行，删除 38 行

本文只总结最新提交相对分叉基线的最终状态，不展开各中间 commit 的参数演变。文件路径均为仓库根目录下的相对路径。

## 变化文件覆盖表

以下增删数量按 Git diff 统计，包含代码、注释和空行；位置是最新提交 `bddb986` 中的行号。被删除的旧代码已不存在于最新文件，因此位置栏主要标记新增或替换后代码所在位置。

| 相对路径 | 状态 | 新增 | 删除 | 最新文件中的变化位置 | 影响范围 |
| --- | --- | ---: | ---: | --- | --- |
| `.gitignore` | 修改 | 10 | 0 | 208–217 | 训练/推理工作流支撑 |
| `as_baseline/experiment/robotwin_fine_tune.sh` | 新增 | 57 | 0 | 1–57，整个文件 | 训练 |
| `as_baseline/experiment/robotwin_inference.sh` | 新增 | 29 | 0 | 1–29，整个文件 | 推理 |
| `as_baseline/scripts/action_embedder_init_weight_patch.py` | 新增 | 16 | 0 | 1–16，整个文件 | 训练 |
| `as_baseline/scripts/action_embedder_zero_init_weight_patch.py` | 新增 | 28 | 0 | 1–28，整个文件 | 训练 |
| `as_baseline/scripts/data_adapter_robotwin.py` | 新增 | 217 | 0 | 1–217，整个文件 | 训练数据准备，同时为推理准备 annotation |
| `as_baseline/scripts/robotwin_action_loader.py` | 新增 | 81 | 0 | 1–81，整个文件 | 推理 |
| `as_baseline/scripts/train_local_checkpoints_launcher.py` | 新增 | 50 | 0 | 1–50，整个文件 | 训练 |
| `cosmos_predict2/_src/imaginaire/trainer.py` | 修改 | 46 | 0 | 56–57、152–191、344、348–350 | 训练 |
| `cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py` | 修改 | 6 | 1 | 159–163、165 | 训练/推理的 checkpoint 获取支撑 |
| `cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow.py` | 修改 | 93 | 0 | 659–747、774–777 | 训练/推理共用实验配置 |
| `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py` | 修改 | 69 | 33 | 68、138–143、180–195、300–302、306–309、311–316、318–330、332–335、338–353 | 训练 |
| `cosmos_predict2/_src/predict2/checkpointer/dcp.py` | 修改 | 22 | 0 | 195–211、217、579、589、602、704 | 训练 checkpoint |
| `cosmos_predict2/action_conditioned.py` | 修改 | 9 | 4 | 315、354–357、365–368 | 推理 |

## 影响训练的变化

### `.gitignore`

**代码量与位置：** 新增 10 行、删除 0 行；最新文件第 208–217 行。

### `as_baseline/experiment/robotwin_fine_tune.sh`

**代码量与位置：** 新增 57 行、删除 0 行；第 1–57 行，整个文件均为新增。

- 提供 RoboTwin 14D action-conditioned 模型的最终训练入口。

### `as_baseline/scripts/action_embedder_init_weight_patch.py`

**代码量与位置：** 新增 16 行、删除 0 行；第 1–16 行，整个文件均为新增。

- 用于 Action Encoder 全随机初始化的对照训练。

### `as_baseline/scripts/action_embedder_zero_init_weight_patch.py`

**代码量与位置：** 新增 28 行、删除 0 行；第 1–28 行，整个文件均为新增。

- 定义 zero-init 版本的 `RobotwinActionChunkConditionedDiT`。
- 两个 Action Encoder 分支都使用：
  - `fc1`：随机初始化。
  - `fc2.weight`：全零。
  - `fc2.bias`：全零。

### `as_baseline/scripts/data_adapter_robotwin.py`

**代码量与位置：** 新增 217 行、删除 0 行；第 1–217 行，整个文件均为新增。

- 将 RoboTwin HDF5 episode 转换为 Cosmos action-conditioned JSON annotation。

### `as_baseline/scripts/train_local_checkpoints_launcher.py`

**代码量与位置：** 新增 50 行、删除 0 行；第 1–50 行，整个文件均为新增。

- 在运行原始 `scripts.train` 前注入三项本地运行适配：
  - Cosmos-Reason URI 命中时直接返回 `COSMOS_REASON_CHECKPOINT_PATH`，避免重复下载。
  - `COSMOS_TEXT_ENCODER_RANK0_LOAD` 开启时，仅 rank 0 从磁盘加载 text encoder state dict。
  - annotation 中的绝对视频路径失效时，退化为 basename，再交给配置的数据集根目录解析。
- 最后使用 `runpy.run_module("scripts.train", run_name="__main__")` 进入原训练流程。

### `cosmos_predict2/_src/imaginaire/trainer.py`

**代码量与位置：** 新增 46 行、删除 0 行；第 56–57、152–191、344、348–350 行。

- 新增 `SIGUSR1`、`SIGTERM` 和 `SIGINT` 的优雅停训处理。
- signal handler 只记录请求，不直接执行 CUDA、collective、日志或 checkpoint I/O。
- 每个 optimizer update 完成后，通过分布式 `all_reduce(MAX)` 将任意 rank 的停训请求同步到全部 rank。
- 收到请求后退出训练循环，并沿用训练器结尾的最终 checkpoint 保存与 finalize 流程。
- 手动停训推荐使用 `kill -USR1 <pid>`。

### `cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py`

**代码量与位置：** 新增 6 行、删除 1 行；第 159–163 行新增代理处理，第 165 行替换 `uvx` 命令参数。

- Hugging Face CLI 下载 checkpoint 时检查 `ALL_PROXY` 或 `all_proxy`。
- 代理协议为 `socks://`、`socks4://`、`socks5://` 或 `socks5h://` 时，自动给 `uvx` 增加 `--with socksio`。
- 避免通过 SOCKS 代理下载训练或推理 checkpoint 时因缺少 SOCKS 依赖而失败。

### `cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow.py`

**代码量与位置：** 新增 93 行、删除 0 行；第 659–747 行定义实验，第 774–777 行注册实验。

- 新增并注册 `ROBOTWIN_14D_FINETUNE` 实验及其 debug 变体。
- 继承 Cosmos Predict2.5 2B reason-embedding action-conditioned rectified-flow 配置，并切换到 action-chunk 网络。

### `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py`

**代码量与位置：** 新增 69 行、删除 33 行；变化位于第 68、138–143、180–195、300–302、306–309、311–316、318–330、332–335、338–353 行。

- `Dataset_3D` 新增可选 `action_dim` 参数。
- 不再固定使用单臂 7D action，而是从首个 annotation 推断机械臂数量：
  - 每臂 6D state。
  - 每臂 1D gripper。
  - 每臂 7D action。
- 校验 state 维数、gripper 维数和 action 维数之间的一致性。
- action scaler 按机械臂数量复制；双臂使用 14D scaler。
- `_get_robot_states` 保留所有机械臂状态，不再只截取前 6 维。
- 普通逐帧相对 action 和 `accumulate_action` 两条路径都改为逐机械臂独立计算。
- 保持对原有单臂 7D 数据的兼容。

### `cosmos_predict2/_src/predict2/checkpointer/dcp.py`

**代码量与位置：** 新增 22 行、删除 0 行；第 195–211 行增加 Gloo group helper，第 217、579、589、602、704 行将其接入 DCP 读写。

- 创建并缓存一个包含全部 rank 的 Gloo process group。
- 模型、optimizer、scheduler、trainer state 的 DCP 加载，以及 DCP 保存，都显式使用该 group。
- 单卡、distributed 不可用或尚未初始化时仍使用默认行为。
- 目的是让分布式 checkpoint 的规划和协调避开 NCCL 相关限制。

## 影响推理的变化

### `as_baseline/experiment/robotwin_inference.sh`

**代码量与位置：** 新增 29 行、删除 0 行；第 1–29 行，整个文件均为新增。

- 提供 RoboTwin 5-task zero-init checkpoint 的推理入口。

### `as_baseline/scripts/robotwin_action_loader.py`

**代码量与位置：** 新增 81 行、删除 0 行；第 1–81 行，整个文件均为新增。

- 为推理读取双臂 RoboTwin action。
- annotation 已包含 `action` 时直接读取；否则根据 12D state 和 2D gripper 重建 14D 相对 action。
- 按左右臂分别应用：
  - 位姿 action scaler。
  - gripper scaler。
- 读取原始视频的 `start_frame_idx` 作为初始 conditioning frame，并按配置调整分辨率。

### `cosmos_predict2/action_conditioned.py`

**代码量与位置：** 新增 9 行、删除 4 行；变化位于第 315、354–357、365–368 行。

- 允许最后一个不足完整 chunk 的 action 序列补零后继续推理。
- 记录最后一个 chunk 的真实 action 数量，只输出真实 action 对应的视频帧。
- 不再把 padding action 生成的帧反馈为下一帧或写入结果视频。
- 拼接多个 chunk 时删除后续 chunk 的首个重复 conditioning frame，保留其最后一个真实生成帧。
- 修复 chunk 边界重复帧和不完整尾块产生多余帧的问题。

### `cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow.py`

- 推理通过同一个 `robotwin_14d_finetune` 实验恢复正确的 2B action-chunk 网络结构。
- 最终结构参数为 14D action、80-step chunk、时间压缩率 4、`state_t=21` 和 480×640 分辨率。

### `as_baseline/scripts/data_adapter_robotwin.py`

- 转换后的 annotation 同时是推理输入。
- `action` 字段和 `robotwin_action_loader.py` 使用相同的左右臂 14D 排列。
- `videos`、state、gripper、task text 和 episode metadata 为推理提供输入视频及条件信息。

### `.gitignore`

- `as_baseline/inference_params.json`、转换后的 checkpoint 和推理输出属于本地实验产物，默认不进入 Git。
- 因此运行时推理参数可能随本地工作区变化，不属于 `a2c298b..bddb986` 的已提交 diff。

## 最终训练参数

以下参数是最新提交中的实验配置与 `as_baseline/experiment/robotwin_fine_tune.sh` 命令行覆盖合并后的最终值。

| 参数 | 最终值 |
| --- | --- |
| 实验 | `robotwin_14d_finetune` |
| 项目 | `cosmos_predict2_action_conditioned` |
| Job group | `robotwin` |
| 默认 job name | `robotwin_14d_finetune_pretrained_4gpu_5task_zero` |
| 基础模型 | Cosmos Predict2.5 2B action-chunk rectified-flow |
| Action Encoder 默认初始化 | `fc1` 随机、`fc2` 全零 |
| 可选初始化 | `ACTION_EMBEDDER_INIT_METHOD=random` 时两层全随机 |
| action dim | 14 |
| action 排列 | `[left xyz, left rpy, left gripper, right xyz, right rpy, right gripper]` |
| action chunk | 80 |
| 输入像素帧数 | 81 |
| temporal compression ratio | 4 |
| `state_t` | 21 |
| 分辨率 | 480×640 |
| FPS downsample ratio | 1 |
| 位姿 action scaler | 20 |
| gripper scaler | 1 |
| 数据集根目录 | `/data1/fangxuebin/cosmos-predict2.5/as_baseline/converted_dataset4` |
| 训练 annotation | `converted_dataset4/train/annotation` |
| 验证 annotation | `converted_dataset4/val/annotation` |
| 每 rank batch size | 4 |
| GPU 数 | 默认 4 |
| 名义全局 batch size | 16，不含继承配置中可能存在的梯度累积 |
| 并行方式 | FSDP |
| FSDP shard size | 4 |
| optimizer | `fusedadamw` |
| 学习率 | `5e-5` |
| weight decay | `0.1` |
| max iteration | 10000 |
| checkpoint save interval | 2000 |
| EMA | 开启 |
| 恢复 checkpoint | `.../robotwin_14d_finetune_pretrained_4gpu_5task_zero/checkpoints/iter_000003062` |
| load training state | `True` |
| strict resume | `True`，由启动脚本覆盖实验配置 |
| object store load/save | 关闭 |
| straggler detection | 关闭 |
| 训练中采样 callback | regular/EMA 两个 callback 均删除 |
| master port | 12345 |
| Cosmos-Reason | 本地 snapshot `375e24000b24baed78f4618d3dd779e47cd96323` |
| text encoder 加载 | 默认仅 rank 0 从磁盘加载 |
| VAE | 本地 Cosmos-Predict2.5 2B snapshot 中的 `tokenizer.pth` |
| 输出根目录 | `/data1/fangxuebin/models/` |
| 训练日志 | `train_zero_5task.log` |

## 最终推理参数

启动脚本来自已提交的 `as_baseline/experiment/robotwin_inference.sh`。逐样本参数来自当前工作区的 `as_baseline/inference_params.json`；后者被 `.gitignore` 忽略，不属于最新 commit，但它是当前脚本实际引用的运行时配置。

| 参数 | 最终值 |
| --- | --- |
| 实验 | `robotwin_14d_finetune` |
| 模型 checkpoint | `.../robotwin_14d_finetune_pretrained_4gpu_5task_zero/checkpoints/iter_000004000/model` |
| 转换后 checkpoint | `as_baseline/checkpoints/robotwin/robotwin_14d_finetune_pretrained_4gpu_5task_zero_iter4000/model.pt` |
| context parallel size | 1 |
| 建议 GPU 数 | 1 |
| 输入根目录 | `as_baseline/converted_dataset4/test` |
| annotation 子目录 | `annotation` |
| 处理范围 | `[0, 100)` |
| camera id | 0 |
| action loader | `as_baseline.scripts.robotwin_action_loader.load_robotwin_bimanual_action_fn` |
| action dim | 14 |
| action chunk | 80 |
| single chunk | `True` |
| start frame index | 0 |
| FPS downsample ratio | 1 |
| 位姿 action scaler | 20.0 |
| gripper scaler | 1.0 |
| state key | `state` |
| gripper key | `continuous_gripper_state` |
| rotation representation | Euler，`use_quat=False` |
| 分辨率 | 480×640 |
| guidance | 0 |
| denoising steps | 35，JSON 未覆盖，使用代码默认值 |
| latent conditional frames | 1 |
| prompt | 空 |
| negative prompt | 空字符串 |
| seed | 0；当前 single-chunk 从 action index 0 开始，实际生成调用 seed 也是 0 |
| reverse | `False` |
| 输出 FPS | 30 |
| JSON save root | `outputs/robotwin_14d_finetune_pretrained_4gpu_5task_zero_iter4000` |
| CLI output directory | `outputs/robotwin_14d_finetune_pretrained_4gpu_5task_zero_iter4000` |
