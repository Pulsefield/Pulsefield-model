# Pulsefield：给定 time skeleton 的因果谱面续写
## 下一阶段实现计划书

- 日期：2026-09-15
- 状态：实施草案；本轮任务与边界按用户反馈确定，数值配置为起始配置而非已经验证的最优值。
- 本轮交付：可以从真实谱面前缀开始、逐行更新全部历史特征与缓存、在固定时间骨架上生成合法非空完整 rows 的可训练系统。
- 实现参照：公开 `witness-style-probe` 分支 `a7ab19f26cd8dd05480c8376060aaced1af4f2a7`。用户报告的后续本地 relation-matching 实验不作为新主干的前置或默认模块。

## 1. 本轮要解决的问题

将原来的“可见前后文条件下的 masked action reconstruction”改为“给定时间骨架和已提交历史的 causal continuation”。当前行只能利用它之前已经给定或生成的 actions。生成一行后，这一行必须成为后续预测的真实历史，更新 occupation、动作时钟、局部 time-action 摘要、note relations 和 temporal memory。

这一轮共同设计三个相互依赖的部分：局部因果摘要、可控制资源消耗的长程建模、窗口采样与整体序列学习目标。不是继续给旧 masked-reconstruction 模型追加独立微调选项。

成功首先意味着获得正确、可训练、可完整 rollout 的新 baseline；预测质量与生成质量分别报告，不能以一项微小 NLL 改善代替整个交付，也不能保证新架构一定胜过某个旧模型。

### 范围冻结

本轮包含：原始谱面解析与合法前缀；三层 causal time-action convolution；causal note-relation attention；有界 long-range temporal attention；joint-row prediction 与实际随机 sampling；新的 training-window sampler 和 sequence objective；真实谱面 teacher-forced evaluation 与 free-running generation；完整输出和资源验证。

本轮不包含：audio conditioning、time skeleton 预测、插入/删除/移动时间点、style/evidence/action reader、style 控制与 demand 标定、RL、diffusion、beam-search 系统、relation-matching 微实验、optimizer/dtype 大规模搜索。不要把这些旧研究任务或未来任务重新挂回主路径。

保留数据身份、song-group split、V3 action 语义和 legality；不沿用 near/detailed/coarse views、双向 action context、旧 ActionReader、按 block 重置的 decoder memory 或旧 equal-block mean-row loss。

## 2. 生成合同：先明确输入、目标和边界

### 2.1 预测对象

给定严格递增、已去重的时间骨架

\[
\Gamma=(t_1,\ldots,t_N),
\]

每个时间点必须产生一个完整四轨非空 row：

\[
a_i\in\{\mathrm{EMPTY},\mathrm{TAP},\mathrm{LN\_START},\mathrm{LN\_CLOSE}\}^4
\setminus\{(0,0,0,0)\}.
\]

骨架包含原始 attack 与 release 时间的并集；模型不知道待预测 row 原来是 attack、release 还是两者混合。非空指“至少一个 lane action”，不指“至少一次 press”：release-only row 是合法生成结果。固定事件数不等于固定 TAP/LN_START 数量。

给定 seed prefix `H_k`，目标是

\[
p_\theta(a_{k+1:N}\mid H_k,\Gamma)
=\prod_{i=k+1}^{N}
 p_\theta(a_i\mid H_{i-1},t_{\le i},e_i),
\]

其中 `e_i` 是调度器提供的真实 skeleton 终点标志；历史在 teacher forcing 时由已经消费的真值构成，在 rollout 时由已经采样的输出构成。有限 neural memory 是对完整历史的实现性摘要，不宣称保留任意久远历史的所有信息。

### 2.2 对“只使用历史”的严格解释

本轮采用严格的在线 feature 合同：完整 skeleton 可以由调度器保存，但模型预测第 i 行时只获得当前时间和此前时间，不使用 following gap、未来局部 pace、未来 source-event 类型、未来 note count 或未来 action 派生信息。调度器允许在真实 skeleton 终点提供 `is_terminal`，用于完整谱面 LN closure；这是显式合法性条件，不是未来 action 信息。

已知未来时间可否用于模型，是以后单独的任务变更。本轮不悄悄引入。不要输入 target 窗口归一化位置、采样窗口的终点、annotation section 边界、source 标识或歌曲标题。计算分块本身不能改变预测。

尤其要隔离原始 `.osu` 中已知的 LN endpoint：给定 LN_START 只意味着 lane 已占用、开始时间已知，不意味着它在未来何时关闭已知。原始 endpoint 可留在 target/source owner 中，不得进入模型的 prefix feature、关系边或 neural cache。

### 2.3 “至少前 30 个 note”采用明确的计数约定

本计划把 note 计作一个原始 hit object，即一次 TAP 或 LN_START；LN_CLOSE 不额外计数，一个四键 chord 计四个 notes。取第一次累计达到至少 30 个 notes 的完整 simultaneous row 作为最短 seed 终点，该 row 不能拆开。此前的 release rows 同样 replay。完整历史从谱面开头开始；不是在任意局部窗口内伪造“新谱面的前 30 个 notes”。

因此，seed 可能多于 30 个 notes，也可能少于 30 个 event rows。必须同时记录 `seed_note_count`、`seed_row_count`、实际时长。没有合法 target 后缀的短谱面要明确标为不适用于本任务，而不是偷偷降低门槛。

### 2.4 中间窗口与真实谱面终点

中间训练或生成窗口结束时允许 LN 保持打开，不能因为资源 chunk 或 sample horizon 结束而强制 LN_CLOSE。

完整 skeleton 的最后一行必须：关闭全部仍打开的 lanes；禁止新的 LN_START；原本关闭的 lanes 可 TAP 或 EMPTY；整行仍非空。这在 V3 单 lane action 规则下有合法候选，不需要新增骨架时间点或在导出时补隐式 release。

已有 seed 中尚未闭合的 LN，要在生成后缀中完成其 endpoint；不得沿用原始文件的未来 endpoint 来完成输出。

## 3. 单一在线状态机：训练与生成必须走同一语义

概念上采用以下边界，名字可适应工程规范，但职责不可合并成隐式副作用：

```text
prefill(seed_rows) -> State
predict(State, current_time, terminal_flag) -> JointRowDistribution
sample(distribution, SamplingPolicy, RNG) -> CompleteRow
commit(State, current_time, CompleteRow) -> NewState
```

`predict` 不读取当前 target，不推进历史，不因被重复调用而改变缓存。`commit` 恰好提交一次完整 row，四个 lanes 同时更新。

训练：`predict -> score true row -> commit true row`。
生成：`predict -> sample legal row -> commit sampled row`。

**Teacher forcing 不等于一次性把整个目标窗口的真实排列灌入 feature builder。** 并行 teacher forcing 可以预计算因果张量，但必须与逐行 reference engine 等价；当前及未来真值只能存在于监督分支中。

State 分开维护：

| 状态 | 内容与约束 |
| --- | --- |
| ExactReplayState | 四轨 occupancy、打开 LN 的开始时间、最近 attack/release 时钟、hand/lane clocks；完全由 committed history 决定 |
| LocalState | 三层 causal convolution 的有限 buffers、时间和 validity；缓存属于对应的已提交行 |
| RelationState | 有界最近关系索引、被读取节点的表示、最多四个活跃 LN head；活跃 LN 的精确事实不得因 neural memory 淘汰而遗失 |
| TemporalState | 有界 recent memory、coarse memory、位置/时间与 provenance；不能随整曲长度无限增长 |
| ExecutionState | 下一骨架位置、RNG、版本、资源策略与日志游标；不作为可学习的答案捷径 |

窗口与 chunk 是计算/监督边界，不是重新初始化谱面的语义边界。状态检查点必须连同 RNG 和骨架位置保存；相同状态恢复后，固定 seed 的继续生成应可复现。

## 4. 新主干：三层局部摘要、关系 attention、长程 attention

总体数据流：

```text
committed action history + current query time
        -> 三层 causal time-action 局部摘要
        -> causal note-relation attention
        -> 有界 causal temporal attention
        -> 当前生成状态 h_i
        -> joint-row distribution -> sampling
        -> commit 整行并更新所有历史状态
```

这里的 attention 属于生成主干，不重新接入旧的 multi-level ActionReader，也不增加一个旧式 decoder GRU 来独立承担 prefix modeling。可复用 joint action 表、左右手对称参数化及输出 coupling，但旧 GRUCell history path 不应成为隐藏的第二主干。

### 4.1 三层卷积重新定义为 causal local summaries

第一版保留三层，以减少同时搜索的自由度。建议使用 causal offsets `{0,-d,-2d}`，d 为 1、2、4；在**已提交事件**上形成最多 3、7、15 行的有序局部摘要。当前真实/生成 row 提交后，更新它对应的新摘要；预测下一行时使用这些摘要，而不是在整段 target 开始前只算一次。

核的连接权重必须依赖实际 elapsed time 与已经可见的 endpoint actions/roles；gap 不能仅作为最后附加的一个 scalar。可采用：

\[
z_j^{(\ell)}=z_j^{(\ell-1)}+
F_\ell\!\left(
 \sum_{r\in\{0,d_\ell,2d_\ell\}}v_{jr}
 K_\ell\bigl(r,\phi(t_j-t_{j-r}),a_j,a_{j-r}\bigr)
 \operatorname{LN}(z_{j-r}^{(\ell-1)})\right).
\]

该式只在 `a_j` 已 commit 后计算；`v` 是真实历史有效性。它规定因果支持与条件来源，不要求实现任意 dense dynamic matrix。现有通道调制思路可以保留，但双向 gather、following gap 与未来 action features 必须删除。

每级摘要显式携带有效事件数和实际时间跨度。3/7/15 events 不被宣称为固定毫秒尺度；LN 持续期间的长 gap 不能自动当成“休息、状态归零”。原始 row facts 与 exact state 保留旁路访问，不把所有事实压进一个反复归一化的摘要。

时间特征只使用已发生 gaps 与当前 query gap。需要 pace 时，采用有明确初始化和有效计数的 causal running statistic；不要用未来 gap 修订历史 pace，也不要把 source-event pace 命名为 musical BPM。第一版可复用物理时间和 smooth basis 的数值函数，但不能复用旧双向 `event_geometry`。4ms basis 与 basis 数量搜索不属于当前里程碑。

### 4.2 context 不完整时的初始化

必须区别三种情况：真实谱面开头还没有前驱；有前驱但被 memory policy 截断；batch padding。三者都不能伪装成一段合法的 EMPTY rows。

短历史使用无未来信息的 BOS/boundary representation、有效性 mask、实际可见跨度与 history-status 字段。缺失 gap 用 availability 表达，不编造 0ms gap，也不重复第一行填满卷积。左右手初始化保持共享/镜像约束。

正常训练从真实开头或正确 replay 的 prefix 构造 state。先给定 30 notes 不保证十五个 event rows 都存在，因此短历史初始化仍是必要功能。后续随机窗口需 prefill 全部对应历史或加载精确同版本状态；不能在 sample 起点清空状态。

### 4.3 note-relation attention 必须对未来动作无知

relation 内容来自已提交的 lane/hand attack succession、same-lane recurrence、已发生的 chord 组织、LN_START 与后续已发生的 LN_CLOSE、仍打开的 LN head。

预测当前 row 时还不知道它在哪些 lane attack，因此不能用真实 target lane 选择邻居。采用 candidate-independent 的四 lane/两 hand frontier query，读取每条 lane 的最近若干关系记录。它表达“若此处选择该 lane 的动作，有哪些历史关系可用”，不宣称当前动作已经发生。

同一 row 的双手关系可由 joint output coupling 表达；该 row 提交后才能作为完整事实加入历史关系。新事件不能反向改写旧时刻已缓存的 causal representation。

邻居数量固定上限，按已声明的关系类别保留最近若干个。活跃 LN head 单独 pin，不能因很久以前开始而使合法性依赖丢失。不得构造整曲 dense adjacency，也不得用未来 close 身份提前连边。

### 4.4 long-range temporal attention：有界 recent + coarse memory

采用带跨 chunk memory 的 causal temporal attention，不对整曲展开 full attention。近历史保留细粒度事件状态；更远历史用有限个按时间排序的 coarse tokens。coarse token 只由已经结束的历史区间构造，保存区间起止、计数与压缩状态，不包含未提交后缀。

起始方案：recent 保留最近 512 个 source-event row slots；每 16 个退出 recent 的已提交行形成一个 coarse token；最多保留 64 个 coarse tokens，超出后按声明规则淘汰。该结构提供有界远程摘要，并不等于无限全历史精确 attention。不同 layers/heads 的 KV 预算都要算入总量。

query 应同时保留事件顺序与实际时间差。不能只根据 event position 假设均匀采样，也不在这一轮加入未声明的 BPM adapter。

训练中 chunk 内允许反传，chunk 之间的 neural memory 使用明确的 stop-gradient；forward 信息继续传播，但不声称跨任意长时间有完整梯度。inference 使用同一 memory 更新/淘汰规则。归档、合并与 evict 必须按事件进度发生，不能因某次 padding 或 batching 改变可见历史。

## 5. 资源合同：把“不随整曲长度 OOM”做成架构性质

### 5.1 起始资源配置

以下是面向现有 24 GiB Apple Silicon 环境的保守起始点，须实测而不是视作安全证明：

| 项目 | 起始值 / 硬边界 |
| --- | --- |
| hand hidden width | 128，左右手共享算子 |
| temporal blocks / heads | 2 / 4 |
| local layers | 3 |
| training microbatch | 2 |
| differentiable chunk Q | 128 source-event rows |
| recent memory M | 512 rows |
| coarse memory S | 64 tokens |
| relation neighbors K | 每 hand query 最多 64 个；active LN facts 另有四轨固定上界 |
| dtype | 先保持 FP32；本轮不以 mixed precision 掩盖无界分配 |
| parsed-source cache | 有界，起始最多 16 个 |
| MPS driver / process RSS guard | 可沿用 8 GiB / 12 GiB 起点，分别实测，不能相加解释为物理内存 |

若按两手的两组 token 计算，主要 attention 工作区上界为

\[
O\bigl(B\,L\,2H\,Q(M+S+Q)\bigr),
\]

持久 neural memory 为 `O(B L (M+S) d)`，局部 buffers 与 relation indices 也有固定上界。整曲长度只能增加流式计算时间和落盘数据，不得增加 GPU attention 维度、活跃计算图或缓存列表的长度。

不把 FlashAttention/CUDA fused backend 当作 MPS 上成立的假设。普通 SDPA/math fallback 的实际 workspace 也必须符合预算。

### 5.2 必须落实的保护

长 target 按 Q 分块，分块反传并释放图；跨 chunk 保留 detached history state。不能为最终 sequence loss 保存整段 tensor graph，不能把每行完整 logits 无限 append 在设备上。统计和生成文件流式落盘。

同一目标窗口的 chunks 保持参数版本不变，累积完整窗口的梯度后再 optimizer step；一批 sampled windows 之间可做有界 gradient accumulation。下一窗口用更新后的参数重放 prefix。第一版不复用跨 optimizer 版本的 learned prefix cache；原始解析结果和 exact replay 数据可以版本化复用。

执行前做最大支持配置的 forward/backward 和长 rollout 实测，包含高密度、超长 LN、长静默间隔与长谱。请求超出已验证配置时提前拒绝或在运行前确定更小 microbatch；不要中途静默缩短语义 context、改变 M/S、删除候选 rows 或丢掉某些高密度样本。

OOM guard 不是“永不 OOM”的证明。目标是：在声明并验证的硬件/负载包络内，内存随谱长保持有界并有足够余量；无法对任意其他进程占用、驱动异常或未知输入给绝对保证。发生资源异常须保存可恢复状态并显式终止，不能用 EMPTY、teacher forcing 或截短输出掩盖失败。

## 6. training-window sampling 与序列 objective 共同定义

### 6.1 两种 sampling 必须分开命名

`WindowSamplingPolicy` 选择训练谱面、prefix 和 target horizon。
`DecodeSamplingPolicy` 从当前合法 joint-row distribution 选择实际输出。

两者有独立配置、seed、日志和实验身份，禁止把输出 temperature 与训练样本权重混为一谈。

### 6.2 training sampler 的起始分布

沿用防泄漏的 song-group split，先按 song group、再按 beatmap 选择；annotations 不改变某首谱面的 pretraining 权重。

对每张谱面找出第 30 个 note 所在的完整 seed row。按 target 起点之前额外可用的历史事件数分三个 context strata：种子之后 0–63、64–511、512 及以上。先在可行 strata 中等概率选择，再均匀选择对应的 target 起始 event。全部边界在训练前固定；不能根据 hidden target 的 loss/style/recurrence 选择起点。

target 采用实际时间 horizon，第一版使用 1s、4s、16s 三个配置化尺度等概率选择。target 为从所选起始事件起、落入该半开时间范围的全部 source-event rows；至少包含起始 row；跨真实谱面终点则截至真实终点并提供 terminal 合同。参数仅是起始尺度，不宣称是音乐语义边界。

每个 target 前的真实完整 prefix 都需因果 prefill，至少含最初 30 notes。prefill 可以 no_grad，不能带入未来信息。第一版接受这项明确的计算成本；若后续优化，必须保持相同 memory 构造与参数版本，不可偷偷只 replay 最近 30 notes。

16 秒高密度 target 超过 Q 只增加 chunks，不缩短 target。记录整个抽样路径概率、prefix notes/rows/elapsed time、有效 recent/coarse coverage、target 时间与行数、terminal 标志、warm-up 费用。短谱缺少某些 strata/horizon 的再分配规则必须可复现。

### 6.3 新主 loss：完整 continuation code length，固定尺度归一化

对 sample w 定义

\[
S_w=-\sum_{i\in I_w}\log p_\theta(a_i^\star\mid H_{i-1}^\star,t_{\le i},e_i).
\]

主风险采用

\[
\mathcal L_{\mathrm{seq}}=
\frac{1}{B Z}\sum_{w=1}^{B} S_w,
\qquad Z=128\ \text{作为固定数值参考尺度}.
\]

Z 不是每个窗口自己的行数，也不是当前 microbatch 的随机 token 数。因此长窗口拥有更多总训练权重，窗口内的一次错误不会因为所在窗口长而被额外除以该长度。这个取舍必须和 1/4/16s 的抽样预算一起承认；不能宣称同时实现了等窗口、等 row、等时间三个不相容的目标。

chunk losses 按同一个 `B*Z` 分母累积，不能“每 chunk 求均值后再平均”，最后一个短 chunk 也不能获得额外权重。optimizer 在约定的 accumulation endpoint 更新，gradient clipping 只针对该次完整累积梯度应用。

要求在参数、history state、memory policy 和 terminal 条件相同的情况下，sequence cost 对纯计算分块可加：

\[
S([a,c))=S([a,b))+S([b,c)\mid\operatorname{Replay}(H,A_{[a,b)})).
\]

相同 prefix 下改变 loss window 的划分不能改变该行预测。TBPTT 会改变跨 chunk 的梯度路径，不得把这种前向/计分一致性写成与全 BPTT 梯度完全相同。

### 6.4 context 少时不确定性更大：如何反映

不要求 short-context 与 long-context 获得相同 NLL，不将较少 context 下对唯一真值的偏差都当作实现错误。训练预算由事先选择的 context strata 定义；概率模型通过保留较宽分布表达不确定性。

禁止用本 batch loss 的倒数、按窗口拟合的 entropy 或动态“难度分数”自动减轻难例。评估按 context strata、target duration、rollout position 分开报告概率校准、code length/rate 与生成结构；需要时使用只由训练集拟合的简单合法 prefix prior 作相对预测基线，但它不作为启动实现的前置。

### 6.5 一个有限、可关闭的结构辅助项

本轮不引入 style reader 或 RL reward。实现 history-conditioned structural marginal loss，直接从同一个 joint-row 分布求和，不增加独立“质量判定器”。

令 `f_r(a,H)` 为一项只依赖候选 row 和已知历史的结构事件：例如当前 press/chord/hand 配置；最近 attack 次序中的 same-lane return / intervening-lane return；LN_START/LN_CLOSE 与当前 occupation 的组合。它们是形式特征，不等同于 Jack/Trill/Tech 的定义。

对真值对应的结构值 `y=f_r(a_i^*,H)`：

\[
P_{\theta,r}(y\mid H)=
\sum_{a\in\mathcal A_i:\,f_r(a,H)=y}p_\theta(a\mid H),
\quad
\ell_{r,i}=-\log P_{\theta,r}(f_r(a_i^\star,H)\mid H).
\]

候选只有至多 255 个，可精确枚举这些 marginals。使用归一化的固定特征组权重，定义

\[
\mathcal L=\frac{1}{BZ}\sum_w\left[S_w+
\lambda_{\mathrm{struct}}\sum_{i\in I_w}\sum_r\eta_r\ell_{r,i}\right].
\]

保留 `lambda_struct=0` 的可训练基线；非零值为显式候选配置，不能自动做广泛搜索。这个辅助项强调特定结构事件的预测，不假设“重复越少越好”“变化越平滑越好”，不把所有合法替代谱面标成坏样本。它仍不是对整条生成轨迹质量的保证，实际长段组织必须由 free-running 输出检验。

训练初版保持 teacher forcing。不要将 generated prefix 与未调整的原始 suffix 真值直接拼接计算 CE；生成改变 occupation 后，源真值甚至可能不再合法。scheduled sampling、反事实重标和 policy-level 训练均不在本轮默认范围内。

## 7. 输出必须是 SamplingPolicy 驱动的完整生成

输出头保留合法 joint-row 概率，不做四条 lane 的独立抽样。主要接口不是“返回 logits”，而是“给定 State 和当前时间，产生一个完整合法 row、更新后的 State 与可追溯的抽样记录”。

顺序固定：

1. 根据真实 prefix occupancy 构造合法 support，去掉 all-empty，并应用真实终点 closure 条件。
2. 得到原始模型 log probabilities，另行应用有版本的可选 history-only prior。
3. 应用 temperature 和 nucleus/top-p，在剩余合法 support 内重新归一化并采样。
4. 提交 row，更新所有 exact/learned state，进入下一个 skeleton 时间点。

一般形式：

\[
\pi(a\mid H)\propto\mathbf1[a\in\mathcal A_i]
\exp\{(s_\theta(a,H)+\beta b(a,H))/\tau\},
\]

再进行已声明的 nucleus 截断。起始生产路径可取 temperature=1、top_p=0.95、beta=0；同时保留不截断的原始模型 sampling 作为评估参照。greedy 是调试对照，不能成为唯一 rollout。

允许以后接入从训练谱面拟合的 chord/return/LN 行为软 prior；当前接口须区分其贡献，但不把拟合这个 prior 变成本轮主干的前置。任何 prior 都只能看生成历史和当前时间；不能硬禁止 Jack、Trill、重复、爆发，也不能把未经标定的“人类极限”写成 chart legality。

记录 raw model log probability 与实际 decode-policy probability，不能混报二者为同一个 NLL。nucleus 或可选 soft filter 必须保证至少保留一个合法候选；数值异常不能以原谱真值、EMPTY 或静默修改 LN 来兜底。

free-running 阶段在 seed 之后不再读取真实 action，不重新贴回真值上下文。输出可以跨任意计算窗口持续生成，并保留 LN carry。

## 8. 真实谱面验收，而非只看汇总 NLL

### 8.1 正确性合同

必须覆盖：当前/未来 target 更换不影响更早 logits；future LN endpoint 不泄漏；训练 teacher forcing 与逐行 replay 一致；完整 row 同时更新；unknown/truncated/BOS/padding 区分；镜像 equivariance；prefix/skeleton 无未来 action side channel；窗口分块无前向语义变化；checkpoint 恢复；nonempty/legality/terminal closure；实际 output schema。

dense/step parity 在固定参数、eval mode、相同 memory 更新规则下比较。关系邻居、有效性与 exact state 要先严格一致，再允许规定的浮点输出容差。

### 8.2 输出级交付

从真实 held-out song groups 中固定一组 continuation cases，覆盖稀疏/密集、长静默、反复 lane return、hand alternation、chord 变化、LN carry/release 与真实终点。使用实际 source 片段，而非仅 toy fixtures；可复用人工确认的典型片段作展示，但不将其重新变成 style supervision。

每例交付原始前缀与后缀、teacher-forced 逐行代价、至少多个随机 seed 的完整续写、greedy 对照、时间对齐渲染、exact state/抽样摘要。生成 `.osu` 或可无损导出的 row 文件，并用独立 replay 检查。生成相对原谱的 lane 选择不同不自动是错误。

### 8.3 评估分层

概率质量：sequence NLL、nats/row、prefix-context strata、target duration、生成位置，以及相同 seed 条件下的 calibration/合法先验对照。

生成结构：press/chord 分布、lane/hand recurrence、短 motif 的持续/转换、LN duration/occupation、长段退化和多样性。比较真实后缀与多次生成的分布，不用一个“全谱完全匹配率”代替。

资源：prefill 与 decoding 分开计时，rows/s、每行延迟分布、峰值设备/RSS、随谱长增长的 live memory、cache 命中/淘汰范围。

旧 bidirectional masked-reconstruction 的 NLL 不能直接与新 causal task 排名。首次整体改版比较回答“新 baseline 是否可训练、能否实际续写、表现为何”，并不拆分归因每个模块。当前旧 query-matching 候选不作为需要再次战胜的门槛。

## 9. 里程碑与依赖

| Milestone | 实现内容 | 完成条件 | 不允许替代成交付 |
| --- | --- | --- | --- |
| M0：因果数据与 state 合同 | skeleton、30-note seed、ExactReplay、pre/post state、终点规则 | 真实谱面逐行 teacher replay 与 prefix 构造正确；未来信息隔离 | 再写一份旧 run audit |
| M1：在线主干 | 三层 causal local summary、note-relation attention、有界 temporal attention、joint head；移除旧 reader/GRU prefix 路径 | 同一 prefix 的 batch/step 行为一致；提交输出确实改变后续所有相关 features | 只验证一个 relation score 反例 |
| M2：sampling 与 sequence training | 新 WindowSamplingPolicy、完整 sequence code length、可关结构 marginal loss、chunked backward/prefill | 能在真实不同 history/时长条件上训练；分块不改变风险权重；无未来 action | 继续沿用旧等 block mean-row loss |
| M3：实际生成与资源封装 | DecodeSamplingPolicy、持续 rollout、LN carry、导出、缓存与内存保护 | seed 后纯生成；长谱合法非空、终点闭合；资源在声明包络内有界 | 只输出 logits 或 teacher-forced accuracy |
| M4：真实 corpus run 与交付 | 新任务训练曲线、固定 case outputs、概率/结构/资源报告 | 交付可复现 run、checkpoint、完整生成文件与明确局限 | 用 300-update 小差值宣告总体方向成败 |

M0 是基础；M1 与 M2 的设计需要共同对齐，不要求先获得旧任务的显著收益；M3 的 sampling/资源语义在 M0 即确定，不能最后补丁式修 LN；M4 使用已完成的同一系统。这里的 milestone 是一条主任务，不是五个可以分别“无收益、停止采用”的微型研究项目。

运行预算由真实吞吐与目标监督事件量共同设置，必须统计 prefill 成本，不能只用 update 数量表达训练暴露。软件 smoke 可以很短；性能 run 的解释必须与其训练量匹配。先得到一套完整新 baseline，再决定是否需要分解 ablation，不先展开 architecture×sampler×loss×optimizer 网格。

## 10. 工程迁移边界与结束标准

建议在独立的 `research/oracle_time_continuation`（名称可调整）中建立新的合同与 runner，保留旧实验可复现，不用一串开关让双向 masked 与 causal continuation 共享隐含 assumptions。

可复用：原始 `.osu` admission、四动作表、ExactReplay 基础、hand-role mapping、source split/identity、数值 time basis、部分 joint coupling、资源日志与渲染/导出工具。

必须重写或重新审核：prefix feature builder、pace/relations、三层卷积 gather、temporal 主干、旧 decoder GRU/ActionReader 接口、window sampler、loss aggregation、generation API、snapshot schema。

旧 checkpoint 不做 exact resume。需要迁移 embedding 等可兼容参数时，只允许 weights-only initialization，记录来源，optimizer/memory/sampler 使用新状态；首个 baseline 可直接从头训练，不将迁移变成前置。

结束标准是：代码能用一份完整配置重现实验，固定真实前缀可生成完整合法谱面；所有 action-dependent features 来自实际已提交历史；长时间运行没有随谱长扩张的活跃内存；sequence risk 与两种 sampling 的含义清楚；真实案例能展示学到了什么及仍失败在哪里。质量结果允许不理想，但不能用不相关小实验替代这些交付。

## 11. 防跑偏检查

每个 PR/阶段说明都回答一个问题：它怎样推进“给定时间骨架、仅凭已提交历史进行真实逐行续写”？无法回答的改动不进入本轮。

以下行为视为偏离本计划：重新接入 future action context；先按真值整窗建 relation graph 再 masking；种子之后偶尔塞回真值修复生成；把 batch/chunk 起点当作真实 BOS；在中间窗口关闭所有 LN；以长 target 费内存为由偷偷缩短语义窗口；把 hidden action 用于图选边；重新接入 style/action reader；以 generic repetition penalty 禁止谱面重复；用独立 toy 表达力测试或微小 aggregate NLL 差异代替完整续写交付。

## 参考依据与身份

[R1] Pulsefield-model，`docs/formulation/notation.md`，公开提交 `a7ab19f26cd8dd05480c8376060aaced1af4f2a7`：完整 rows、四动作合法性、LN closure、committed-prefix 合同。

[R2] 同提交，`src/pulsefield_model/research/source_action_modeling/{local_representation,representation,time_basis,model,actions}.py`：现有双向局部支持、pace、ActionReader/GRU 与 action schema。本文提出的 causal path 不是这些代码已经实现的功能。

[R3] 同提交，`src/pulsefield_model/configs/hydra/source_action_composition.yaml`：现有 MPS 环境与资源 guard 起点。

[R4] Dai et al. (2019), Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, ACL P19-1285 / arXiv:1901.02860。这里只借用跨 segment 的有界 memory 与 stop-gradient 思路，不移植论文效果结论。

[R5] PyTorch 官方文档，`torch.nn.functional.scaled_dot_product_attention`、`torch.mps.set_per_process_memory_fraction`、`torch.mps.current_allocated_memory`，2026-09-15 检索：backend 与 memory accounting 需按实际硬件验证。

[R6] Huszár (2015), How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?, arXiv:1511.05101：scheduled sampling 的统计目标并非自动保留标准 likelihood 意义。本文同时有独立的 chart-legality 理由，禁止未调整的生成 prefix/真值 suffix 混接。
