# Timing v3 决策记录：真实 ground truth 与 `.osu` note-grid 证据

日期：2026-08-17  
状态：已接受，取代“合成变速语料作为最终验收 proxy”的解释

## 决策摘要

1. 先前生成的 identity、persistent jump 和 linear ramp 音频只保留为机制压力测试。其解析 time map 可以验证模型是否恢复了**注入的**变化，但不能代表真实歌曲中的自然 tempo 结构，也不能作为 Timing v3 的最终质量验收。
2. `.osu` redline 通常包含有价值的制谱判断，很多时候也接近正确；但它带有人类 BPM、offset、tactus 和局部边界误差，因此只能作为带不确定性的 proposal、软先验和诊断，不能逐值当作 ground truth。
3. `.osu` note onset 是 mapper 根据听感、节奏意图、可玩性和反复游玩反馈留下的节奏观测。即使 redline 有偏差，note 与潜在 beat grid 的关系仍有参考价值。后续 `.osu` 的主要用途从“复现 redline”改为“约束潜在连续 beat axis”。
4. 最终验收必须回到未做人工 time-warp 的真实录音。权威 reference 应优先保存逐拍时间、downbeat、拍号、可接受 metrical level 和不确定区间；局部 BPM 和变化形态应从连续 beat-time reference 派生，而不是预先强制成 constant/jump/ramp 三类。

## 对合成因果语料的重新定级

stable256、fresh128 和此前 causal288 的变换标签在数学上没有错误：生成器确实知道施加的 time map、倍率和 seam。错误发生在结果解释层——解析变换真值曾被提升成真实音乐任务的 ground truth。

这些语料仍可回答：

- 模型是否对已知播放速率变化有响应；
- 表示、序列化和 seam 计算是否连续；
- 算法是否会遗漏一个受控、持久的变化；
- 实现对倍率方向、边界和长音频是否稳定。

它们不能回答：

- 原歌曲的绝对 BPM、拍点或自然 tempo curve 是否正确；
- 真实演奏中的 rubato、ritardando、accelerando、tactus 切换或剪辑应属于哪一类；
- 模型是否学习了 phase-vocoder、固定倍率集合或固定 seam 范围的人工痕迹；
- 产品在自然音乐上的 constant/change 质量是否达标。

因此，fresh128 即使完成并通过，也只能记录为受控 time-warp 外测，不再解锁产品质量结论。

## `.osu` 证据的新口径

### Note onset

对一个候选连续 beat axis `b(t)`，将每个 note onset `t_i` 映射到 beat coordinate `b(t_i)`，衡量它到合理音乐细分格的距离。证据应按 audio identity 聚合所有难度，防止同一音频因谱面数量较多而重复计权。

允许的细分不能只写死为四分或八分；应覆盖三连音以及常见的 `1/4`、`1/8`、`1/12`、`1/16`、`1/24`、`1/32` 等格点。`1/32` 本身不等于错误。“乱”、装饰音、少量 mapping 错误、LN 末端和特殊 pattern 应由 robust likelihood、局部支持和跨难度一致性降权，而不是通过单一细分阈值删除。

Note-grid 证据表达的是“这个 beat axis 能否解释 mapper 实际放置的节奏事件”，不是“谱面提供了精确物理 BPM”。

### Redline

Redline 可以：

- 提出 BPM family、phase、offset 和变化边界候选；
- 给出 mapper 认为合理的节拍层级和局部结构；
- 用于 note-grid 解释力、跨难度一致性和音频结果的事后诊断。

Redline 不可以：

- 作为要求模型逐毫秒复现的唯一标签；
- 在存在 half-time、double-time、低整数 tactus 或小 BPM 误差时直接判错；
- 单独决定真实 tempo change 是否存在。

推理与评价必须允许全局 offset 小修正、BPM 小误差、alias-aware tactus，以及边界附近的不确定区间。音频、note-grid 和 redline 三路证据应分别报告，不应通过同一来源的重复变换制造“独立多数票”。

## 真实 reference 的优先级

最终 benchmark 应优先使用：

1. 与确切录音对齐、经过人工检查的 beat/downbeat 时间；
2. 乐谱或 performance MIDI 与真实录音的人工校准对齐；
3. 两名以上标注员的逐拍校正和分歧裁决，并保留 ambiguous/metre-level alternatives；
4. 对项目自身 5,050 首音频建立 source-disjoint、录音去重的人工标注集。

公开数据可以作为训练、开发或外部诊断来源：ASAP 覆盖古典钢琴的强 rubato 和渐变速度；RWC/AIST 提供流行、古典、爵士等真实录音的人工 beat structure；POP909 提供流行歌曲的人工 tempo alignment，但部分 audio beat 标签来自算法；Mazurka Project 提供窄曲目、多演奏版本的逐拍 tempo。使用当前 BeatThis frontend 时，ASAP、RWC、Hainsworth、SMC、Harmonix 等已进入其训练数据的集合不能再充当独立最终外测。

## 后续研究边界

- 停止把合成三分类的准确率写成真实歌曲准确率。
- 不因本决策立即重跑或揭盲 fresh128；现有冻结产物保留作机制审计。
- 下一条主研究问题改为：在不把 redline 当真值的前提下，音频证据与跨难度 note-grid 证据能否共同恢复真实录音的连续 beat axis。
- 在真实人工 reference 建立前，full5050 上的结果只能称为弱监督训练、内部一致性或审计结果，不能称为 ground-truth benchmark。

## 决策分类

- `KILL`：合成 identity/jump/ramp 作为最终验收 proxy。
- `MUTATE`：将 `.osu` 从 redline 拟合源改为 note-grid 弱监督；redline 降为带误差 proposal。
- `PENDING TEST`：真实录音 note-grid/audio 联合恢复需要新的 Experiment Card；本记录不授权实现或运行。
