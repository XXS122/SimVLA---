# 研究方案总览（Task Proposals Index）

五个创新点的论文级设计文档。每份文档独立成篇，含：科学问题、逐篇相关工作对比（附链接）、
形式化方法、红队自我批判、分阶段实验设计（可行性预实验 → 主实验 → 消融 → 机制分析）、
预注册可证伪结论、审稿人攻击模拟与投稿策略。

| # | 文档 | 一句话定位 | 风险 | 总算力(GPU天) | 建议目标 |
|---|------|-----------|------|--------------|---------|
| P1 | [P1-staleness-world-model-rl.md](P1-staleness-world-model-rl.md) | 陈旧感知的世界模型在环 RL 后训练 | 高 | 300–400 | NeurIPS 2027（暂缓） |
| P2 | [P2-energy-verified-onestep-flow.md](P2-energy-verified-onestep-flow.md) | 能量校验一步流 + 流式 VLA 推理扩展定律 | **低** | 40–70 | **ICLR 2027（主攻）** |
| P3 | [P3-fisher-nullspace-insulation.md](P3-fisher-nullspace-insulation.md) | VLA 共训练梯度干扰的测量-理论-方法三部曲 | 中 | 100–150 | NeurIPS 2027 / TPAMI |
| P4 | [P4-merge-aware-cotraining.md](P4-merge-aware-cotraining.md) | 零通信双站点可合并协同训练 | 低-中 | 80–120 | ICML 2027 |
| P5 | [P5-counterfactual-invariance.md](P5-counterfactual-invariance.md) | 状态配对反事实重渲染 + 不变性正则 | 中 | 60–100 | CoRL 2027 / ICLR 2028 |

## 深度调研后的关键结论（相对第一版摘要的修正）

1. **P2 出现了两个此前未列入的强竞品**：V-GPS（CoRL 2024，价值函数重排生成式策略动作）
   和 Mean-Flow one-step VLA（arXiv 2603.01469）。P2 文档已重新划定创新边界：
   贡献收缩为"生成器-校验器协同设计 + 极值理论解释幂律 + 无奖励标注的对比校验器"三点，
   删去"首个一步流 VLA"的表述。结论：**仍然成立，但必须抢时间**。
2. **P3 的零空间投影方法本身已不新颖**（GNSP, arXiv 2507.19839 已做 VLM 持续学习版本）。
   P3 文档将定位从"提出零空间方法"整体迁移到"**同时性共训练**（而非顺序持续学习）的
   干扰谱测量与 Fisher 理论"，方法退居第三贡献。结论：**成立，但叙事必须以测量和理论为主**。
3. **P4 需要正面处理 ATM**（arXiv 2411.03055，交替微调-合并）与
   "去中心化学习中单次全局合并即足够"（arXiv 2507.06542）这两篇。P4 的护城河收缩为
   "通信代价为显式约束的异构 VLA 设定 + 合并损耗上界 + 硬件即实验平台"。结论：**成立**。
4. **P1、P5 无直接撞车**，但 P1 算力不可行（维持暂缓），P5 有被定性为
   "domain randomization++"的审稿风险（文档中给出三层防御）。

## 建议执行顺序

```
现在 ──► P2（主攻 ICLR 2027，代码已就绪，见 docs/TASK2_PIPELINE.md）
   └─► P2 的 SFT 教师训完后，A800 空闲期启动 P4 的 Phase 0 预实验
P2 投稿后 ──► P4 主实验（ICML 2027）＋ P3 Phase 0 测量实验
P3 测量结果发表为 workshop/短文 ──► P3 完整版（NeurIPS 2027）
P5 视 P2/P4 结果与人力启动；P1 待算力扩容后重估
```

所有文档遵循同一实验纪律：每个阶段设 **go/no-go 判据**，预实验不通过就砍方向，
不为沉没成本追加算力。
