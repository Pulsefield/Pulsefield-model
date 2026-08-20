# Timing v3 研究时间线

日期：2026-08-16  
状态：实验暂停；fresh128 未揭盲；2026-08-17 已撤销合成变速作为最终验收 proxy

## 当前结论

Timing v3 已经完成四次重要转向：

1. 从“拟合 `.osu` redline”转向“区分 redline、note-grid、音频估计和真实录音 reference”；redline 不再是硬标签，但 note 落点仍是重要的人类节奏证据。
2. 从“继续扩大全局搜索”转向“先用真实音频确定失败发生在倍率、候选覆盖还是边界排序”。
3. 从“围绕 ACF 候选堆特征”转向“用整曲 BeatThis 前端表示直接预测边界”。
4. 从“合成 identity/jump/ramp 充当最终验收”转向“合成变速只做机制压力测试，最终验收必须使用未变换真实录音上的逐拍 reference”。

当前最强的受控 time-warp seam 结果来自完整 stable256 开发集上的 source-disjoint 8 折 OOF：`217/256` 首在 `1 s` 内，中位误差 `107 ms`。它证明模型能识别 phase-vocoder 注入边界，但不再被解释为真实歌曲 tempo-change 质量，也不能作为最终通过。

## 时间线

### 2026-08-11：先修评价基础，再谈模型提升

- Experiment 001 在 5,050 个音频 identity 上重建了可恢复的评估基础，所有统计先按音频分组，避免同一歌曲的多个难度重复计权。
- `.osu` redline 被降为 mapper 注释和弱比较器。它可表达制谱意图，但不能直接充当物理音频真值。
- object grid 很适合验证 phase，却几乎不能识别 half-time、double-time 等 tempo alias。后续必须把“相位一致”和“倍率选择”拆开报告。

**方向变化：** 研究目标不再是复现某张谱面的 timing points，而是从音频生成一个可序列化、全局连续的 beat axis，并把 mapper 一致性放在单独的评价层。

### 2026-08-11：相位连续表示成立，早期投影算法不成立

- Experiment 002 建立了以绝对 beat index 表示的 constant/jump section。相邻 section 共享同一个累计 beat 轴，jump 只能改变导数，不能重置 phase。
- 该表示和序列化连续性被保留，但 Family B 的平均 phase guard 略超上限，因此算法被拒绝。
- Experiment 003 的联合 anchor/BPM 投影修复了 phase 指标，却在 holdout100 出现 `9/100` fallback，超过 `5%` 上限。

**方向变化：** 表示层和推理层正式分离。phase-continuous schema 是可复用成果，某个投影器通过局部指标并不等于产品可接受。

### 2026-08-11 至 08-13：全局搜索、局部分解和协议冻结

- Experiment 004 的全曲 constant/jump 搜索即使做了 exact-equivalent 加速，长曲仍无法满足运行预算。扩大 beam、候选或 timeout 不再被视为可行主线。
- Experiment 005 的局部 frontier 虽保留了正确多段候选，冻结目标函数仍偏向跳过中间 tempo 的捷径。
- Experiment 006 用 boundary-conditioned 左右 tempo evidence 修复了合成机制，`44/44` synthetic arms 通过，但没有真实音频结论。
- Experiment 007 只冻结了协议、来源绑定、恢复和验证器。2026-08-13 的完成审计明确判定 Phase 1 尚未完成，真实 holdout、broad、full5050 和生产集成都没有通过。

**方向变化：** 停止把 synthetic pass 当作音频质量证据。接下来每个结论必须说明它来自协议测试、自然音频弱证据，还是已知变换的因果真值。

### 2026-08-14 至 08-15：候选生成与保留系列找到局部机制，但没有泛化

- Experiment 014 至 021 逐步定位 short-ABA 失败：候选在 cap 前存在，随后被 family imbalance 和 scalar retention 剪掉。Pareto reservation 最终在两条已暴露机制样本上恢复正确 ABA，同时保住 stable constant。
- Experiment 022 扩到 42 条后失效：row runtime p90 `11.724 s`，stable 出现 6 个 false jump，jump boundary recall 很低。
- Experiment 025 的完整生命周期审计显示，20 条 jump 中多数 direct jump evidence 在生成前就不存在，或在 retention 前被剪掉；只有 1 条 direct candidate 最终被选择。

**关键结论：** 两条机制样本上的候选修复不能外推到真实歌曲。继续微调 retention、eligibility 或 selector 只会在同一失败面上移动误差。

### 2026-08-15：重新定义“真值”和 BPM 口径

- full5050 多难度审计显示，大多数同组谱面直接复制 timing grid，同组一致性主要证明复制和同一 set 内约定，不能证明跨 mapper 或物理真值。
- BeatThis 与 mapper nominal BPM 的比较强烈支持 octave tempo family，但不足以唯一决定 direct tactus。half-time、double-time 和低整数倍率必须作为潜变量处理。
- 对 phase 的口径也被拆开：alias-family 上的最近 beat lattice 属于可听物理一致性；继承 source beat 编号的 parity 只是坐标约定。后者不能作为音乐真值门槛。
- `.osu` 的毫秒 offset 确有约 `0.5 ms` 量化下限，但这只解释小误差，不能解释数秒级边界错误或倍率混淆。

**方向变化：** mapper agreement 全部降格为弱证据。后续主门改用对 waveform 施加已知连续 time map 得到的因果真值。:codex-annotation{index="1"}

### 2026-08-15 至 08-16：真实 454 首审计暴露 tactus 混淆

- balanced454 中，可确认的 15 个 stable false jump 全部符合相同模式：raw、BeatThis 都认为全曲稳定，但 selector 把同一物理 tempo family 内的 `2x`、`1/2x`、`3/2x` 等 tactus 状态切换解释成真实 BPM jump。
- false jump 的 MDL gain 与真阳性大量重叠，增加单一阈值或 BIC 惩罚无法同时保住 stable specificity 和 material recall。
- TempoCNN 24 模型 ensemble 走向另一个极端：454 条中只预测 1 条 change，material recall 几乎归零。
- source-union bridge 在这批弱审计上达到 stable `33/33`、observation contained `54/134`、selected precision `0.783`。它证明多源音频区间有定位信号，但这些区间仍是同一 waveform 上的相关估计，不是独立物理真值。

**方向变化：** “直接 BPM 候选 + 阈值”被替换为“物理 tempo family 与 tactus state 分层”的建模思路；同时决定用 causal warp corpus 检验真实变化，而不是继续在 mapper/estimator 共识上优化最终分数。

### 2026-08-15 至 08-16：运行时从主要阻塞降为次要阻塞

- full454 profile 找到最大热点是重复 phase-origin/grid scoring，不是 DP 本身。
- request-local memoization、物理曲线时间轴向量化、DP first-argmin 向量化、global candidate exact 优化和 raw schedule 缓存都保持冻结输出逐行一致。
- 最终 real600 单进程完整链路达到约 `4.973 s`，进入 `5 s` 目标以内；full454 exact replay 保持 `454/454` 输出一致。

**关键结论：** 运行时已经有一条 exact-equivalent 可行路径。当前优先级应回到质量，尤其是 change class、倍率和 seam，不应再为了局部毫秒收益改变模型语义。

### 2026-08-16：production causal288 基线确认质量尚不可发布

- 当前 production 路径在 96 identity、96 jump、96 ramp 的已知变换语料上得到：constant `81/96`，jump `29/96`，ramp `0/96`。
- jump seam 在 `1 s` 内仅 `4/96`，倍率与 seam 的联合命中仅 `1/96`；ramp 没有被选择的生产 lane。
- phase continuity 为 `288/288`，说明表示和装配没有断，但曲线识别本身失败。

**方向变化：** 不再把 full454 的弱标签改善当作接近产品可用。接下来围绕 source-disjoint causal corpus 单独解决 ratio 与 seam，再回到 class/ramp 集成。

### 2026-08-16：建立 stable256 开发集和 fresh128 外测集

- stable256 由 256 个真实稳定音频源构成，每源生成 identity、persistent jump、linear ramp，共 768 条。train/holdout 各 128 个 source，四种倍率和 seam 位置平衡。
- route ID 改为不可推导的随机 ID；train truth、holdout truth、assignment 和生成器物理隔离。初版可逆 route ID、combined truth 和公开 state side channel 都在任何模型消费前修复。
- fresh128 再选 128 个与 stable256 source-disjoint 的真实音频源，生成 384 条 opaque routes。truth 目录 mode `000`，截至暂停时从未揭盲。

**关键结论：** 这是目前最可靠的评价层。它只证明模型对已知相对 time warp 的响应，不提供歌曲绝对 BPM 或自然曲线真值。

### 2026-08-16：ACF 能找对倍率和候选峰，但 global argmax 排错峰

- 全曲 ACF dilation 在 stable holdout128 上，selected ratio 有 `125/128` 落在 `2%` 内，说明倍率估计已相对可靠。
- 直接取 ACF global argmax 时，holdout seam 只有 `61/128` 在 `1 s` 内。
- 真 seam 最近的 ACF local maximum 在 holdout 上 `128/128` 位于 `1 s` 内；top16 oracle 为 `118/128`，完整 dev256 的 top16 oracle 为 `238/256`。
- signed error 没有稳定方向偏置，误差也不随 duration、seam fraction 或倍率方向稳定变化。固定时间补偿和窗口群延迟解释被排除。

**方向变化：** seam 的主要问题是候选排序，不是 100 ms 网格、renderer 映射、倍率方向或候选覆盖。

### 2026-08-16：候选排名系列接近门槛，但停止继续堆特征

- 9 维 top16 线性 ranker 把 holdout `61/128` 提升到 `77/128`，paired gains/losses 为 `22/6`，说明排序可以学习。
- 581 维 rich linear 在完整 dev256 的 8 折 source OOF 达到 `199/256`，中位误差 `423 ms`，但低于门槛 `205/256`。
- 固定浅层 MLP 为 `197/256`，按倍率方向拆 head 为 `199/256`，扩到 top32 为 `194/256`。top32 oracle 已达 `246/256`，覆盖增加反而降低实际命中。

**关键结论：** 同一候选和特征族的容量、分方向、候选数量调整都没有跨门。该 ranker family 已停止，不能再靠小改动追 6 条样本。

### 2026-08-16：多个独立 seam 机制被排除

- complex phase lattice、dynamic IOI change point、固定倍率 segmental local-tempo 都未超过 ACF baseline。
- paired local-rate Conv1d 在 holdout 仅 `3/128` 首进入 `1 s`，预测普遍滞后约 `4.4 s`，局部 rate trace 接近常数。
- post-transformer hidden 特征的 OOF 为 `194/256`，且静态审查发现它由约 30 秒 global-attention chunks 拼接，含人工 reset seam，因此结果不再作为可靠证据。

**方向变化：** 停止从局部 onset、IOI 或拼接后的 Transformer hidden 间接推 seam，改为读取整曲一次前向的 BeatThis frontend 表示，并直接训练平移等变的全轨边界模型。

### 2026-08-16：整曲 BeatThis frontend 边界模型首次通过开发门

- 每条 route 的完整 spectrogram 只调用一次 BeatThis frontend，不经过 main transformer、chunk split 或 hidden stitch。
- 固定模型为 route 内归一化 frontend 序列、audio-only ACF ratio conditioning、无绝对时间输入的非因果 dilated Conv1d；使用同一 8 折 source OOF、固定 50 epochs、无 early stop 和无参数搜索。
- dev256 结果为 `217/256` 在 `1 s` 内，超过 `205/256` 门槛；`100/250/500 ms` 内分别为 `125/196/210`，中位误差 `107 ms`。
- 四种倍率的 `1 s` 命中为 `57/64`、`55/64`、`51/64`、`54/64`。尾部仍重，p90 为 `2.520 s`，最大错误超过 `100 s`。

**结论：** 这是当前唯一达到开发门的 seam 模型。它证明整曲 frontend 的时序变化包含因果 warp 边界信息，但尚未通过外部数据，也没有解决 constant/jump/ramp 三类联合选择。

### 2026-08-17：撤销合成三分类最终验收，重新启用 `.osu` note-grid 证据

- stable256、fresh128 和 causal288 的解析标签只描述生成器注入的 time map。它们适合验证 causal-warp sensitivity、实现连续性和边界恢复，但不能代表真实音乐中的自然 tempo 结构。
- identity 只表示没有额外注入 warp，不表示原歌曲恒定 BPM；persistent jump 和 linear ramp 也不等价于真实演奏中的 tempo change、rubato 或渐变。
- `.osu` redline 通常具有参考价值，但人类标注可能带有小 BPM、offset、tactus 和局部边界误差，因此保留为 proposal、软先验和诊断，不能逐值作为 ground truth。
- 广泛游玩的谱面 note onset 是 mapper 经过听感、节奏意图、可玩性和社区反馈后留下的观测。即使 redline 有偏差，跨难度 note 与潜在 beat grid 的关系仍可为连续 beat axis 提供弱监督。
- 后续 note-grid 模型必须容纳三连音及常见 `1/4`、`1/8`、`1/12`、`1/16`、`1/24`、`1/32` 细分；不能把 `1/32` 本身等同于“乱”。异常 pattern、LN 末端和少量 mapping 错误应通过 robust likelihood 与跨难度一致性降权。
- 真正最终 reference 应来自未变换真实录音上的逐拍/downbeat 人工标注、乐谱或 performance MIDI 对齐，以及明确保留的不确定区间。局部 BPM 和变化形态从 beat-time curve 派生，不预先强制成三类。

**方向变化：** `KILL` 合成 identity/jump/ramp 作为最终验收 proxy；`MUTATE` 为音频与 `.osu` note-grid 联合恢复连续 beat axis，redline 降为带误差 proposal。详细口径见 [2026-08-17 ground-truth 与 note-grid 决策](./timing_v3_decision_2026-08-17_ground_truth_and_osu_note_grid.md)。

## 暂停点

fresh128 one-shot 外测已按要求中止，当前状态如下：

- final dev256 模型已按相同配置训练并冻结。
- 384 条 opaque route 的 audio-only ACF ratio 已全部计算并冻结。
- 整曲 frontend cache 已完成 `118/384` 条，还剩 `266` 条。
- 384 条最终 seam predictions 尚未生成，因此 fresh truth 没有被打开。
- `sealed/` 仍为 mode `000`，外测没有发生任何质量反馈或参数回调。
- 运行进程已确认退出，现有缓存可用于续跑。

2026-08-17 的解释修正后，fresh128 不再是产品最终验收门。若未来恢复，它只能完成受控 time-warp 机制审计；通过或失败都不能直接证明真实歌曲质量，也不能解锁生产发布。当前不应为最终验收目的揭盲，主研究优先级改为真实录音 reference 与 `.osu` note-grid/audio 联合恢复。

## 已稳定的结论

1. `.osu` redline、mapper nominal BPM 和 source beat numbering 都不是物理真值；redline 可作为带误差 proposal 和软先验，不能作为硬标签。
2. `.osu` note onset 是重要的人类节奏观测。应按 audio identity 聚合跨难度 note-grid 解释力，并用 robust likelihood 容纳高细分、异常 pattern、LN 端点和少量 mapping 错误。
3. 合成 identity/jump/ramp 只有注入变换真值，只能用于机制压力测试，不能作为真实歌曲最终验收。
4. octave tempo family 通常可从音频稳定恢复，direct tactus 仍有不可辨识性。倍率和 phase 指标必须提供 alias-aware 版本。
5. 全局 beat axis、phase-continuous section 和无缝序列化是已验证的表示成果，识别质量不足不应推翻表示层。
6. stable false jump 的主要机制是 tactus state 被误当成物理 tempo change，单一阈值无法解决。
7. ACF 的倍率与局部峰覆盖较强，global peak 排序较弱；该结论目前只在受控 time-warp 语料上成立。
8. 继续扩 beam、候选数量、top-k、线性特征或浅层 MLP 已没有充分依据。
9. 当前最值得保留的质量方向是整曲、无 chunk seam 的 pretrained frontend 时序表示；当前最值得保留的性能方向是 exact-equivalent memoization 和向量化。
10. ramp 仍没有通过任何生产质量门；合成 ramp 结果也不能授权真实 ramp 支持。

## 关键记录

- [任务定义](./timing_v3_task_definition.md)
- [问题与决策日志](./timing_v3_problem_log.md)
- [Phase 1 完成审计](./timing_v3_phase_1_completion_audit.md)
- [full5050 音频共识实验](./timing_v3_experiment_027_full5050_audio_consensus_constant_sources.md)
- [stable256 因果语料结果](./timing_v3_experiment_028_result.md)
- [fresh128 外测语料结果](./timing_v3_experiment_029_result.md)
- [2026-08-17 ground-truth 与 `.osu` note-grid 决策](./timing_v3_decision_2026-08-17_ground_truth_and_osu_note_grid.md)
- [ACF 物理基线](../../artifacts/local/timing_v3/acf_dilation_physics_root_v1/result.json)
- [top16 线性排名结果](../../artifacts/local/timing_v3/acf_top16_linear_rank256_v1/result_log.md)
- [rich linear OOF 结果](../../artifacts/local/timing_v3/acf_top16_rich_linear_oof256_v1/result.json)
- [paired local-rate 结果](../../artifacts/local/timing_v3/paired_local_rate256_v1/evaluation.json)
- [整曲 BeatThis frontend OOF 结果](../../artifacts/local/timing_v3/beatthis_frontend_fulltrack_boundary_oof256_v1/result.json)

文中 full454、causal288 和 real600 的数字来自本轮冻结运行记录；其原始结果位于临时实验目录，未作为仓库 source of truth。继续研究前，应把确需复核的临时结果迁入新的受控结果记录，而不是重新扫描全部历史缓存。
