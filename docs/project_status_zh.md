# SimVLA 研究项目:状态与执行日志

> **这是实时更新文档**:每完成一步就在"详细时间线"末尾追加一条,并刷新
> "当前状态与待办"。格式固定:做了什么/为什么 → 执行的命令 → 得到的结果 → 结论/下一步。
>
> 相关文档:
> - `docs/research_proposals_zh.md` —— 最初的 5 个创新方向提案(本项目源自其中"想法一")
> - `docs/experiment_log_zh.md` —— 推理时缩放阶段(实验 1–7)的详细分析版记录
>
> 最后更新:2026-07-06(第二轮 DPO 结果与训练日志诊断:数据多样性是瓶颈,需扩大采集)

---

## 项目地图(一眼看全局)

| 阶段 | 状态 | 一句话结论 |
|---|---|---|
| 想法一:推理时算力缩放(多采样 + 免训练选择) | ✅ 已判死 | 纯采样不提升成功率;但副产品(步数结论、故障探针)有效 |
| 裁决实验:失败是否可挽救 | ✅ 已完成 | 失败是"状态陷入"、系统性的,不可靠采样挽救 → 转 Flow-DPO |
| 想法四:Flow-DPO(从失败中学习) | 🔄 进行中 | 第一轮训练把模型训坏(标签噪声);第二轮已修复问题、待评测结果 |

---

## 详细时间线

### 1. 项目启动:五个创新方向提案(2026-07-02)
**做了什么**:通读代码库架构(SmolVLM-500M 骨干 + 流匹配动作 Transformer + LIBERO 数据流),结合双机算力(SAPI A100 / YYK A800),提出 5 个具顶刊潜力的方向:①推理时缩放,②MoE+终身学习,③流式记忆,④Flow-DPO 偏好对齐,⑤物理信息流匹配。
**产出**:`docs/research_proposals_zh.md`(commit `cc1dce7`)
**结论/下一步**:用户选择先深入做**想法一(推理时缩放)**。

---

### 2. 想法一阶段 0:推理侧代码实现(2026-07-02)
**为什么**:验证"让模型在难时刻多算一点"是否提升成功率,且**不需要重新训练**——直接用现有 ckpt。
**做了什么**:
- `models/modeling_smolvlm_vla.py` 新增 `generate_actions_tts`:批量 best-of-N 采样(共享一次 VLM 前向)、可变去噪步数、三种免训练选择器(consensus/smoothness/first)、不确定性探针、自适应算力塌缩。
- 评测服务/客户端加 TTS 参数透传 + 诊断日志(`diag.jsonl`)+ 结果日志(`results_*.jsonl`)。
- 新增 `run_tts_sweep.sh`(N×S 网格扫描)、`analyze_tts.py`(自动分析)。
**产出**:commit `26e63cb`
**结论/下一步**:进入 `paths.env` 环境适配,然后开始实验。

---

### 3. 机器适配:paths.env 环境变量支持(2026-07-02)
**为什么**:用户机器有自己的路径/GPU 配置,需要脚本零参数适配。
**做了什么**:所有脚本读 `$SIMVLA_SMOLVLM_MODEL`、`$LIBERO_DATASETS`、`$CUDA_DEVICES`、`$NUM_GPUS` 等;修复加载 ckpt 时用本地路径覆盖 config 里存的旧路径(否则离线模式必炸)。
**产出**:commit `a831262`,`paths.env.example`

---

### 4. 实验 1:libero_spatial 上的 N×S 网格(2026-07-03)
**命令**:
```bash
python serve_smolvlm_libero.py --checkpoint <ckpt> --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102 --diag_log ./tts_sweep/diag.jsonl
bash run_tts_sweep.sh 8102 libero_spatial 20 ./tts_sweep consensus
python analyze_tts.py --results_dir ./tts_sweep --diag ./tts_sweep/diag.jsonl
```
**结果**:N=1..16 × S=5,10,成功率 94.0%~97.5%,基线已高(94–96%),天花板效应明显。
**结论**:①spatial 套件测不出 N 轴收益(考卷太简单);②**S=5 全面优于 S=10**(以后默认省一半算力);③共识选择随 N 非单调(苗头,待确认)。

---

### 5. 不确定性探针 bug 发现与修复(2026-07-03)
**发现**:原探针(t=1 处速度场跨种子方差)读数在各 N 下精确等于 `(1-1/N)×噪声方差`——测的是随机噪声本身,和模型状态无关,完全无效。
**修复**:改用一步去噪估计 `x̂₀ = x_t − t·v` 的跨种子方差,数学上精确消掉噪声项。用"对种子不敏感的假模型"验证:旧探针读数 0.96(全噪声),新探针精确为 0。
**结论**:旧日志里的等价字段(`uncertainty_x0`)可直接重新分析,不用重跑仿真。

---

### 6. 实验 3 + 3b:去混杂分析 + K 扫描(2026-07-03)
**命令**:`python analyze_tts.py --results_dir ./tts_sweep --diag ./tts_sweep/diag.jsonl`(同一份日志反复重新分析)
**结果**:
- `r_full`(全程平均不确定性 vs 成败)强负相关 −0.63~−0.82,但 `r_first10`(仅开局)接近 0——**失败并非开局注定**;
- 成功/失败分列后:失败回合全程不确定性是成功回合的 2–3 倍(不止事件附近);
- K 扫描(K=5/10/20/40):**预测性在第 10–20 次决策出现**(≈第一次抓取时段),之后信号饱和。
**结论**:探针的正确定位是"**实时故障检测器**",不是"开局预言家"。

---

### 7. 实验 4:libero_10 决定性网格(2026-07-04)
**命令**:
```bash
N_LIST="1 2 4 8 16" S_LIST="2 5 10" bash run_tts_sweep.sh 8102 libero_10 20 ./tts_sweep_10 consensus
```
**结果**(200 回合/配置):
| N | S=2 | S=5 | S=10 |
|---|---|---|---|
| 1 | 67.5% | 71.5% | 68.5% |
| 2 | 72.0% | 70.5% | 68.5% |
| 4 | 68.0% | 68.0% | 70.5% |
| 8 | 69.0% | 69.0% | 67.0% |
| 16 | 65.5% | 71.5% | (剔除,样本不完整) |

**结论**:①**N 轴正式判死**——65.5–72.0% 无单调性,基线 70% 有 30pp 空间却不涨;②**S=2 足够**(比默认 10 步省 5 倍算力);③新探针 x0hat 全面验证有效(夹爪梯度、成败分离 12/12 一致);④libero_10 开局即可预测(K5 全负)——早期不确定性≈模型自感任务难度。

---

### 8. 实验 5:分叉反事实裁决实验(2026-07-04)
**为什么**:N 轴已判死,核心问题变成——失败到底是"可挽救"(候选里有好动作,选不出来)还是"系统性"(所有候选都不行)?
**做了什么**:新建 `branch_counterfactual.py`:在不确定性飙升(spike 臂)或平静时刻(control 臂,第 15 次决策)存档仿真状态,独立续跑 M 次数成功。修复了 robosuite 步数计数器跨分支不清零导致的报错。
**阈值确定**:
```bash
python -c "... P50 0.00233 P85 0.00763 P90 0.01041 ..."   # 取 P85 = 0.0076
```
**命令**:
```bash
python branch_counterfactual.py --port 8102 --task_suite libero_10 \
    --num_trials 10 --branches 4 --unc_threshold 0.0076 --log ./branch_10/branches.jsonl
```
**结果**:
```
spike:   n=8   mean_branch_success=0.156  all-fail=3/8   all-succeed=0/8
control: n=92  mean_branch_success=0.696  all-fail=10/92 all-succeed=48/92
```
**结论**:control 臂 69.6%≈基线 70%(存档机制可信);spike 臂只有 15.6%(而分支本身就是普通重采样)——**失败是系统性的"状态陷入",重采样救不回来**。

> **🔀 路线切换(用户已确认)**:主线从"想法一·推理时缩放"切换到"**想法四·Flow-DPO**"。

---

### 9. 实验 6:偏好对采集(Flow-DPO 数据引擎)(2026-07-04)
**为什么**:分叉协议天然产出"同状态成功动作 vs 失败动作"配对,扩展成数据采集器。
**做了什么**:`branch_counterfactual.py` 加 `--save_pairs_dir`(保存快照观测 + 各分支首个动作块)、`--randomize_control`(快照点随机撒在第 5–40 次决策)。服务端现在把不确定性一并返回给客户端。
**命令**:
```bash
python branch_counterfactual.py --port 8102 --task_suite libero_10 \
    --num_trials 10 --branches 8 --unc_threshold 0.0076 \
    --randomize_control --control_call 40 \
    --save_pairs_dir ./dpo_data --log ./dpo_data/branches.jsonl
```
**结果**:99 个有效存档,**40 个混合结局**(8 个 spike + 32 个 control)→ 约 160 对(默认每存档限 4 对)。
**附带发现**:本轮 spike 臂分支成功率 58.7%(远高于实验 5 的 15.6%)——采样窗口更宽后捕到的多是"可挽回的岔路口",而非"不可逆深渊",两者是不同性质的不确定性飙升。

---

### 10. 实现 Flow-DPO 训练代码(2026-07-04)
**做了什么**:
- `datasets/dpo_pairs.py`:`DPOPairDataset` 从采集日志枚举(赢家块, 输家块)配对。
- `train_flow_dpo.py`:Diffusion-DPO 的 ELBO 代理适配到速度场——
  `L = −log σ(−β[(d_π(a_w)−d_ref(a_w))−(d_π(a_l)−d_ref(a_l))])`,`d(a)=‖v−u(a)‖²`。
  冻结参考 = BC ckpt 本身(KL 锚防塌缩);VLM 冻结、策略与参考共享一次前向;赢家 BC 项保流形。
  损失数学已单元测试(margin 方向、π=ref 时=log2、梯度只流向策略)。

---

### 11. 实验 7 第一轮训练:训坏了(2026-07-04)
**命令**:
```bash
python train_flow_dpo.py \
    --models ./runs/exp_tds/ckpt-200000 \
    --pairs ./evaluation/libero/dpo_data/branches.jsonl \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --output_dir ./runs/flow_dpo \
    --beta 50 --bc_coef 1.0 --learning_rate 1e-5 --iters 2000 --batch_size 16
```
**训练现象**:`implicit_acc` 早早冲到 1.000(160 对数据被过了约 200 遍,严重过拟合);`margin` 稳定负值;`bc` 项平稳未上涨。
**评测结果**(N=1 S=5,libero_10):**ckpt-dpo-1000 → 52%,ckpt-dpo-2000 → 50%**——两个都比基线(约 63–71.5%,见下方"待澄清事项")明显更差。
**中途 bug**:评测客户端 `log_results` 目录不存在直接崩溃 → 已修复(自动 mkdir)。

**根因诊断**:采集时很多存档是 7/8、1/8 这种**不均衡**结局。对 7/8 成功的存档,那 1 条失败分支的**第一个动作块**被当作"坏动作"喂给模型,但那条分支大概率是**后面几十步才出错**,第一步本身没问题——**标签噪声**。模型被这些错误标注的对训练,把本来会做的动作也带偏了。

---

### 12. 澄清基线数字(未完全解决,2026-07-05)
用户提到:用于所有实验的 ckpt 实际是 2 万步训练的(不是命令示例里写的 200000 步),且该 ckpt 用户自己的评测口径测得 libero_10 成功率 **63%**(与我们 200 回合测得的 71.5% 有约 8.5pp 落差,超出±3.2pp 的采样噪声范围)。
**未完全确认**:该 63% 是否与本项目全程使用的 ckpt 为同一文件、评测协议(回合数等)是否与我们一致。
**当前处理方式**:不纠结精确基线值,以"**目标成功率 ≥70%**"作为后续 DPO 版本的验收线;不管基线是 63% 还是 71.5%,52%/50% 都是明显更差,核心诊断(第一轮训坏)不受影响。

---

### 13. 修复:按分支成功率过滤偏好对(2026-07-05)
**做了什么**:`datasets/dpo_pairs.py` 的 `build_pair_index`/`DPOPairDataset` 新增 `min_branch_rate`/`max_branch_rate`,只保留分支成功率落在中间区间(如 [0.2, 0.8])的存档——这些才是"第一步真正决定命运"的岔路口,过滤掉 7/8、1/8 型的标签噪声。同时支持多份采集日志合并(`--pairs` 可传多个文件)。
**验证**:分支成功率分布(99 个存档):
```
0.00:12  0.12:8  0.25:5  0.38:7  0.62:4  0.75:9  0.88:7  1.00:47
```
`[0.2,0.8]` 区间命中 25 个存档(0.25/0.38/0.62/0.75 那四档);两端的 0.12、0.88(共 15 个,即标签噪声源)被排除。
**过滤后配对数**:`max_pairs_per_snapshot=8` → **200 对**(比过滤前的 160 对还多,因为放宽了每存档上限)。

---

### 14. 第二轮训练:更保守的超参(2026-07-05,进行中)
**改动**(相对第一轮):
- `--min_branch_rate 0.2 --max_branch_rate 0.8`:只用干净的岔路口数据;
- `--beta 50→10`:降低对标签的信任强度;
- `--bc_coef 1.0→3.0`:更强拉住模型贴近原策略,降低出错代价;
- `--iters 2000→500`,`--save_interval 100`:减少过拟合窗口,且**每 100 步存档**,训完从 5 个 checkpoint 里挑最好的,不再赌一个终点。

**命令**:
```bash
python train_flow_dpo.py \
    --models <ckpt> \
    --pairs ./evaluation/libero/dpo_data/branches.jsonl \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --output_dir ./runs/flow_dpo_v2 \
    --min_branch_rate 0.2 --max_branch_rate 0.8 \
    --max_pairs_per_snapshot 8 \
    --beta 10 --bc_coef 3.0 --learning_rate 1e-5 \
    --iters 500 --save_interval 100 --batch_size 16
```
**训练日志关键点**:
```
step 0:   implicit_acc=0.625  margin=-0.006
step 100: implicit_acc=0.875  margin=-0.069
step 260: implicit_acc=1.000  margin=-0.148   ← 首次打满,此后基本稳定在 0.875–1.000
step 480: implicit_acc=1.000  margin=-0.152
```
**结论**:✅ 已完成。`implicit_acc` 明确从 0.625 爬升并在 260 步后饱和于 1.0,
`margin` 持续变负——训练**确实学到了一个自信的偏好方向**,并非被超参压制到没学习。

---

### 15. 自动化多 checkpoint 筛选脚本(2026-07-05)
**为什么**:第二轮会产出 5 个 checkpoint(100/200/300/400/500 步),需要逐个起服务评测再关闭,手动操作容易出错(第一次尝试时"重启服务端"只写成了注释、没真正执行,导致客户端报 `ConnectionRefusedError`)。
**做了什么**:新增 `evaluation/libero/screen_checkpoints.sh`——对每个 checkpoint 自动"起服务(conda run 到 simvla 环境)→ 轮询等待端口就绪 → 跑评测(conda run 到 libero 环境)→ 关服务",最后打印汇总表。
**命令**:
```bash
SIMVLA_ENV=base bash screen_checkpoints.sh \
    ../../norm_stats/libero_norm.json libero_10 10 ./dpo_eval_v2 \
    ../../runs/flow_dpo_v2/ckpt-dpo-100 \
    ../../runs/flow_dpo_v2/ckpt-dpo-200 \
    ../../runs/flow_dpo_v2/ckpt-dpo-300 \
    ../../runs/flow_dpo_v2/ckpt-dpo-400 \
    ../../runs/flow_dpo_v2/ckpt-dpo-500
```
（用户的 simvla 环境实际是 conda `base` 环境,已用 `SIMVLA_ENV=base` 覆盖默认值）

**结果**(2026-07-06,10 回合/任务快筛,100 回合/checkpoint):
| checkpoint | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| 成功率 | 65.0% | 58.0% | 58.0% | 61.0% | 64.0% |

**结论/诊断**(结合本条与实验 14 的训练日志一起看):
- 5 个点在 58–65% 之间无规律摆动(均值 61.2%),每点仅 100 回合(标准误≈±5pp)——**与基线(约 63%)在统计上无法区分**。不是"训得越久越差"(500 步又回到 64%),而是**整条训练轨迹都停留在噪声范围内**,挑其中最高的一点(如 100 步的 65%)是在挑噪声峰值,不能当作真实提升。
- 但训练日志显示 `implicit_acc` 确实爬升并饱和到 1.0——排除了"β/bc_coef 太保守、模型没学到东西"的假设。真正的解释:**训练集只有 25 个不重复状态,模型很可能是在"记住这 25 张画面该模仿哪个动作",而非学到可泛化的规则**——评测时策略走到的是全新画面,训练时学到的"记忆关联"用不上,所以评测端毫无变化(bc_coef 高也起到了保护作用:没见过的状态上策略基本还是原样,故不变差)。
- **裁定**:调超参这条路在当前 200 对数据上已经走到头,真正的瓶颈是**状态多样性太低**。下一步的核心杠杆是**大幅扩大偏好对采集规模**,而非继续微调 β/bc_coef。

---

## 当前状态与待办

**结论**:第二轮(过滤+保守超参)已确认是"空转"——训练学到了自信的偏好方向
(implicit_acc 饱和至 1.0),但因训练集只有 25 个不重复状态而**很可能是记忆而非泛化**,
评测端 5 个 checkpoint 全部落在基线噪声范围内(58–65%,均值 61.2%)。
**裁定**:调超参这条路走到头了,下一步的核心杠杆是**扩大偏好对采集规模**。

**下一步(按顺序)**:
1. **[你]** 扩大采集规模到约 400 回合(4 倍),争取积累 ~100 个干净岔路口状态:
   ```bash
   python branch_counterfactual.py --port 8102 --task_suite libero_10 \
       --num_trials 40 --branches 8 --unc_threshold 0.0076 \
       --randomize_control --control_call 40 \
       --save_pairs_dir ./dpo_data_v2 --log ./dpo_data_v2/branches.jsonl
   ```
   若 YYK 机器空闲,建议并行跑 `libero_goal`(不同任务场景是天然的数据增强,
   比同任务里堆采样更有助于打破"记忆而非泛化"的问题)。
2. **[我]** 采集完成后按同样流程过滤([0.2,0.8] 区间)、用同样保守超参
   (`--beta 10 --bc_coef 3.0`)重训第三轮,重点看**评测是否终于跳出噪声范围**。
3. **[你]** 筛选协议同步升级:下一轮用 `--num_trials 20`(200 回合/checkpoint)
   替代这次的 10 回合快筛——这次的噪声(±5pp)已经证明快筛分辨力不够。
4. **[可选,待办已久]** 把 libero_10 十个任务的 TDS 分数发我 → 做"模型早期不确定性 vs 人工 TDS"相关分析(实验 4 的副产品,免费出图)。

---

## 关键文件清单

| 文件 | 作用 |
|---|---|
| `docs/research_proposals_zh.md` | 最初 5 个研究方向提案 |
| `docs/experiment_log_zh.md` | 推理时缩放阶段(实验 1–7)详细分析记录 |
| `docs/project_status_zh.md` | 本文档:全项目时间线 + 当前待办 |
| `models/modeling_smolvlm_vla.py` | `generate_actions_tts`:批量采样/可变步数/选择器/不确定性探针 |
| `evaluation/libero/serve_smolvlm_libero.py` | 评测服务:TTS 参数、诊断日志、不确定性回传 |
| `evaluation/libero/libero_client.py` | 评测客户端:TTS 覆盖、结果日志、元数据透传 |
| `evaluation/libero/run_tts_sweep.sh` | N×S 网格扫描 |
| `evaluation/libero/analyze_tts.py` | 缩放律表、不确定性对齐(成败分列)、K 扫描 |
| `evaluation/libero/branch_counterfactual.py` | 分叉反事实实验 + 偏好对采集器 |
| `evaluation/libero/screen_checkpoints.sh` | 多 checkpoint 自动筛选 |
| `datasets/dpo_pairs.py` | `DPOPairDataset`:配对枚举 + 分支成功率过滤 + 多文件合并 |
| `train_flow_dpo.py` | Flow-DPO 训练器 |
| `paths.env.example` | 机器本地配置模板 |
