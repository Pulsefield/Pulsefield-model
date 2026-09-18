# Pulsefield：给定 time skeleton 的因果谱面续写

## 下一阶段实现计划书

- 日期：2026-09-17
- 状态：M0–M3 已实现，M4 实测与质量验收进行中；接口与验证范围见[实现说明](oracle_time_continuation.md)。本文保留 M1–M4 的设计与验收合同。生成合同、梯度路径和缓存生命周期是实现要求；模块与数值配置是本轮选定的起始设计，不宣称已经证明其表示充分性或质量最优性。
- 本轮交付：在本机可持续训练、从真实前缀生成完整谱面的系统；给定时间骨架且不使用音频，但生成动作须开始呈现可玩的局部组织与合理的衔接。合法性、可恢复运行和预测损失分别验收，均不能代替生成质量。
- 实现基础：`991d2f987f8a112220f5beb76bcf2d8a0903ade3` 已实现 M0 的 source/seed、exact replay 与 query/commit。继续扩展现有 `research/oracle_time_continuation` owners；旧 source-action 代码只按本文迁移范围复用，relation-matching 不作为新主干的前置或默认模块。

本文的非空事件行、prefix replay 和缓存合同限定这一版 baseline，不是最终目标对时间骨架或架构的约束。其他条件任务及其证据见[时间骨架与学习任务的研究问题](oracle_time_expert_question.md)；改变这些条件时应明确新的预测任务，而不是把本文的实现要求当作不可改变的前提。

## 1. 本轮要解决的问题

将原来的“可见前后文条件下的 masked action reconstruction”改为“给定时间骨架和已提交历史的 causal continuation”。当前行只能利用它之前已经给定或生成的 actions。生成一行后，这一行必须成为后续预测的真实历史，更新 occupation、动作时钟、局部 time-action 摘要、note relations 和 temporal memory。

这一轮共同实现三个相互依赖的部分：局部因果摘要、可控制资源消耗的长程建模、窗口采样与 continuation likelihood。窗口长度、可读取的前向历史、梯度可穿过的历史分别定义；有界保存历史和学会利用历史分别验收。

正确、可训练、可完整 rollout 是工程基础。阶段质量验收还要求真实生成中的动作组织可辨认，LN 与前后动作有可解释的配合，且长段不会退化为无意义的重复或持有。使用 Beatmap Lens 的 Foundation 与当前人工 gold 作组织判断参照；这些标签不等于难度或可玩性分数。预测质量与生成质量分别报告，不能以微小 NLL 改善代替交付。osuT5 与 Mug-Diffusion 是后续比较目标；没有同条件的真实输出比较时，不宣称超过它们。

### 范围冻结

本轮包含：原始谱面解析与合法前缀；三层 causal time-action convolution；causal note-relation attention；有界 long-range temporal attention；joint-row prediction 与实际随机 sampling；新的 training-window sampler 和 sequence objective；真实谱面 teacher-forced evaluation 与 free-running generation；完整输出和资源验证。

本轮不包含：audio conditioning、time skeleton 预测、插入/删除/移动时间点、style/evidence/action reader、style 控制与 demand 标定、RL、diffusion、beam-search 系统、relation-matching 微实验、optimizer/dtype 大规模搜索。不要把这些旧研究任务或未来任务重新挂回主路径。

保留数据身份、song-group split、V3 action 语义和 legality；不沿用 near/detailed/coarse views、双向 action context、旧 ActionReader、按 block 重置的 decoder memory 或旧 equal-block mean-row loss。

## 2. 生成合同：先明确输入、目标和边界

### 2.1 预测对象

给定严格递增、已去重的时间骨架

$$
\Gamma=(t_1,\ldots,t_N),
$$

每个时间点必须产生一个完整四轨非空 row：

$$
a_i\in\{\mathrm{EMPTY},\mathrm{TAP},\mathrm{LN\_START},\mathrm{LN\_CLOSE}\}^4
\setminus\{(0,0,0,0)\}.
$$

骨架包含原始 attack 与 release 时间的并集；模型不知道待预测 row 原来是 attack、release 还是两者混合。非空指“至少一个 lane action”，不指“至少一次 press”：release-only row 是合法生成结果。固定事件数不等于固定 TAP/LN_START 数量。

给定 seed prefix `H_k`，目标是

$$
p_\theta(a_{k+1:N}\mid H_k,\Gamma)
=\prod_{i=k+1}^{N}
 p_\theta(a_i\mid H_{i-1},t_{\le \min(i+r,N)},e_i),
$$

其中 `r` 是有界时间前视行数，`0 ≤ r ≤ 16`；`e_i` 是调度器提供的真实 skeleton 终点标志；历史在 teacher forcing 时由已经消费的真值构成，在 rollout 时由已经采样的输出构成。有限 neural memory 是对完整历史的实现性摘要，不宣称保留任意久远历史的所有信息。

该分解定义主动限制信息访问的模型族，不是关于真实 action 与未来 skeleton 条件独立的结论。Oracle timing 已提供源谱 attack/release 时间并集；此任务的概率质量不能直接代表未知 timing 或 audio-conditioned generation 的质量。

### 2.2 动作因果性与已知时间条件

模型只能读取已提交动作。调度器持有完整 skeleton，并可通过 `model.time_lookahead_rows` 提供最多 16 个后续时间点相对当前时刻的偏移。独立时间编码器读取这些偏移及相邻间隔；它们没有 attack/release 类型、lane、note 数、LN 配对或 annotation 标签。`r=0` 保留无时间前视的对照配置。真实末行的 `is_terminal` 仍是显式合法性条件。

此合同修订了最初禁止 following gap 的限制。完整生成曾在新开 LN 后跨越 77.643 秒无事件空档：未来时间已经属于给定条件，却无法被模型用于开闭决策。允许时间前视使该错误可被学习纠正，不保证网络自动学会，也不额外强制长 gap 关闭。前视始终从完整 skeleton 取值，不依赖 target 窗口、训练 horizon 或计算 chunk。

不要输入 target 窗口归一化位置、采样窗口终点、annotation section 边界、source 标识或歌曲标题。原始 `.osu` 的 LN endpoint 与 head 的配对关系仍归 target/source owner：一个未来 skeleton 时间可能包含 release，但模型不知道它对应哪一条 LN，不能用源谱 endpoint 完成生成 LN。改变未来动作而保持 skeleton 不变，不能影响此前 logits；改变已给定的未来时间则允许影响启用前视的模型。

### 2.3 “至少前 30 个 note”采用明确的计数约定

本计划把 note 计作一个原始 hit object，即一次 TAP 或 LN_START；LN_CLOSE 不额外计数，一个四键 chord 计四个 notes。取第一次累计达到至少 30 个 notes 的完整 simultaneous row 作为最短 seed 终点，该 row 不能拆开。此前的 release rows 同样 replay。完整历史从谱面开头开始；不是在任意局部窗口内伪造“新谱面的前 30 个 notes”。

因此，seed 可能多于 30 个 notes，也可能少于 30 个 event rows。必须同时记录 `seed_note_count`、`seed_row_count`、实际时长。没有合法 target 后缀的短谱面要明确标为不适用于本任务，而不是偷偷降低门槛。

### 2.4 中间窗口与真实谱面终点

中间训练或生成窗口结束时允许 LN 保持打开，不能因为资源 chunk 或 sample horizon 结束而强制 LN_CLOSE。

完整 skeleton 的最后一行必须：关闭全部仍打开的 lanes；禁止新的 LN_START；原本关闭的 lanes 可 TAP 或 EMPTY；整行仍非空。这在 V3 单 lane action 规则下有合法候选，不需要新增骨架时间点或在导出时补隐式 release。

若当前有 $o$ 条 open lanes，普通和 terminal support 分别为

$$
|\mathcal A(o)|=3^{4-o}2^o-1,\qquad
|\mathcal A_T(o)|=
\begin{cases}15,&o=0,\\2^{4-o},&o>0.\end{cases}
$$

普通行最多有 80 个合法候选，可使用固定 256-row 表和 legality mask 实现。这些公式保证存在可完成的续写，不保证末行集中关闭 LN 在组织上合理。Rollout 必须记录进入 terminal 前的 open lanes、各 LN 的已持续时间、末行强制关闭数量与其占全部 LN_CLOSE 的比例，显式展示终点条件的作用。

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

并行 teacher forcing 可以预计算逐时点的 causal content，但必须与逐行 reference engine 等价。对预测 $a_i$ 的 local、relation、temporal 三条输入路径，$a_i$ 及其后的真值都不可见；$a_i$ 只能作为监督，或用于逻辑 commit 后、供更晚位置读取的 content。保护 `PredictionInput` 之外，也必须保护 learned state：不能先用当前 target 建 relation bank，再指望外层 attention mask 修复 residual、query 或 normalization 中的泄漏。

State 分开维护：

| 状态 | 内容与约束 |
| --- | --- |
| ExactReplayState | 四轨 occupancy、打开 LN 的开始时间、最近 attack/release 时钟、hand/lane clocks、累计行数与 note 数；完全由 committed history 决定 |
| LocalState | 三层 causal convolution 的有限 buffers、时间和 validity；缓存属于对应的已提交行 |
| RelationState | 有界 lane 索引、完整 row 节点与关系标签、最多四个活跃 LN head；精确 head 事实与 learned descriptor 分开保存 |
| TemporalState | 有界 recent/coarse layer-input states、位置/时间与 provenance；训练重投影，推理另缓存 K/V；query 无第二套永久 bank |
| ExecutionState | 下一骨架位置、RNG、版本、资源策略与日志游标；不作为可学习的答案捷径 |

窗口与 chunk 是计算/监督边界，不是重新初始化谱面的语义边界。状态检查点必须连同参数/缓存版本、RNG、骨架位置和已持久化输出位置保存；固定 runtime 与执行配置下恢复相同状态，应复现相同的继续生成。

## 4. 新主干：三层局部摘要、关系 attention、长程 attention

总体数据流：

```text
committed action history + current query time
        -> 三层 causal time-action 局部摘要
        -> causal note-relation attention
        -> 有界 causal temporal attention
        -> 当前 pre-row query outputs h_i
        -> joint-row distribution -> sampling
        -> commit 整行并更新所有历史状态
```

这里的 attention 属于生成主干，不重新接入旧的 multi-level ActionReader，也不增加一个旧式 decoder GRU 来独立承担 prefix modeling。可复用 joint action 表、左右手对称参数化及输出 coupling，但旧 GRUCell history path 不应成为隐藏的第二主干。

### 4.1 三层卷积重新定义为 causal local summaries

第一版使用三层 causal offsets `{0,-d,-2d}`，d 为 1、2、4；在**已提交事件**上形成最多 3、7、15 行的有序局部摘要。当前真实/生成 row 提交后，更新它对应的新摘要；预测下一行时使用这些摘要，而不是在整段 target 开始前只算一次。

核的连接权重依赖实际 elapsed time 与已经可见的 endpoint actions/roles；gap 不能仅作为最后附加的一个 scalar。局部更新为：

$$
z_j^{(\ell)}=z_j^{(\ell-1)}+
F_\ell\!\left(
 \sum_{r\in\{0,d_\ell,2d_\ell\}}v_{jr}
 K_\ell\bigl(r,\phi(t_j-t_{j-r}),a_j,a_{j-r}\bigr)
 \operatorname{LN}(z_{j-r}^{(\ell-1)})\right).
$$

该式只在 `a_j` 已 commit 后计算；`v` 是真实历史有效性。第一版固定为 offset-specific 映射加逐通道调制：

$$
K_{\ell,r}(e)=
\operatorname{diag}\!\left(1+\tanh g_{\ell,r}(e)\right)W_{\ell,r},
$$

其中 $e$ 包含该边的 elapsed-time basis 和已提交 endpoint 的相对 hand/role actions。调制因子在 $(0,2)$ 内，只能抑制或放大固定通道映射，不能随条件直接翻转该映射的符号；它是低成本条件核，不是任意动态矩阵，也不构成表示充分性保证。该边界不意味着整个网络不能表达 alternation。动作边仍只连接已提交节点；已知未来时间由独立时间编码器提供，不通过双向 action gather 或未来动作边输入。

每级摘要显式携带有效事件数和实际时间跨度。有效计数按原始 row 支持的并集计算，不能把重叠子摘要的计数直接相加。3/7/15 events 不是固定毫秒尺度；时间调制改变连接权重，不改变所读取的 event 集合。LN 持续期间的长 gap 不能自动当成“休息、状态归零”。原始 row facts、exact state 和各级摘要分别可访问。LayerNorm 仅沿通道，第一版 dropout=0，不引入跨 target 时间统计。

Replay 的时间统计只使用已发生 gaps 与当前 query gap；独立的 skeleton 前视编码遵守第 2.2 节。第一版 pace 采用最近 32 个已完成正 event gaps 的均值，并保留有效 count 与跨度。`predict(t_i)` 单独读取当前 gap $t_i-t_{i-1}$，其 pace 只取已 committed 行之间的 gaps；在 `commit(a_i)` 中将当前 gap 加入一次。重复 predict 不改变统计，首行没有前驱时保持 unavailable。

该均值仅是 event pace。31 个 100ms gaps 加一个约 92s gap 会使均值接近 3s；保留原始 gap、物理时间 basis、count 和 span，不只依赖均值归一化时间，也不称为 local BPM。绝对时间先在 float64 中做差，再转为网络精度；可复用 smooth basis 数值函数，不能复用旧双向 `event_geometry`。4ms basis 与 basis 数量搜索不属于当前里程碑。

`model.time_lookahead_rows` 启用独立的时间 MLP：按事件顺序编码未来偏移与 successive gaps，缺失位置使用 availability 通道。`r=16`、hidden=128 时，它增加 110,848 个参数；共享输出同时加入两手 query/content facts，保持镜像等变。末层零初始化保留已有主干的初始函数，`training.timing_learning_rate` 可为新模块指定独立 AdamW 学习率；所有组共享 warmup、weight decay 和全局梯度裁剪。该路径不增加随曲长增长的 neural cache。

### 4.2 context 不完整时的初始化

必须区别三种情况：真实谱面开头还没有前驱；有前驱但被 memory policy 截断；batch padding。三者都不能伪装成一段合法的 EMPTY rows。

短历史使用无未来信息的 BOS/boundary representation、有效性 mask、实际可见跨度与 history-status 字段。缺失 gap 用 availability 表达，不编造 0ms gap，也不重复第一行填满卷积。左右手初始化保持共享/镜像约束。

正常训练从真实开头或正确 replay 的 prefix 构造 state。先给定 30 notes 不保证十五个 event rows 都存在，因此短历史初始化仍是必要功能。后续随机窗口需 prefill 全部对应历史或加载精确同版本状态；不能在 sample 起点清空状态。

### 4.3 note-relation attention 必须对未来动作无知

relation 内容来自已提交的 lane/hand attack succession、same-lane recurrence、已发生的 chord 组织、LN_START 与后续已发生的 LN_CLOSE、仍打开的 LN head。

第一版固定**完整 row 为 memory 节点**。每 lane 索引最近 12 个 attack 和 4 个 release 的 row IDs，四 lane 共最多 64 个普通索引项；attention 读取这些 IDs 的去重并集。同一 chord 在多条 lane 索引中出现时，仅有一份 row payload，携带全部相应的 lane/role、关系类别与 recency 标签，不因为重复出现在 softmax 中获得额外权重。

预测当前 row 时还不知道它在哪些 lane attack，不能用真实 target lane 选择邻居。每手一个 candidate-independent frontier query，输入保留 outer/inner 的有序区别及四 lane 的已知事实，不先平均两个角色。它表达“若此处选择该角色动作，有哪些历史关系可用”，不宣称当前动作已经发生。容量是首版取舍，不是所有 pattern 已被覆盖的证明。

同一 row 的双手关系可由 joint output coupling 表达；该 row 提交后才能作为完整事实加入历史关系。新事件不能反向改写旧时刻已缓存的 causal representation。

活跃 LN head 另 pin，最多新增四个唯一 row 节点，所以每手实际读取上限为 68。与普通节点同 ID 时合并关系标签，同一个 row 上的多个 open heads 使用 lane mask 表达。精确事实保留 lane、start time 与事件身份；head 的 learned descriptor 是当时的历史描述，不能称为精确 gameplay state 或长期 LN 组织的完整表示。

Corpus 中 LN 最大 event 跨度不是 pin 的上限依据：生成可一直用 EMPTY 保持 occupation，直到普通 token 被淘汰。pin 随真实 commit close 才解除；learned payload 在 TBPTT 边界截断梯度，精确义务不丢失。所有节点保存自己拥有的有限 payload，不持有整曲 tensor view 或无界前驱链。不得构造整曲 dense adjacency，也不得用未来 close 身份提前连边。

### 4.4 long-range temporal attention：有界 recent + coarse memory

采用带跨 chunk memory 的 causal temporal attention。近历史保留细粒度事件状态；更远历史用有限个按时间排序的 coarse tokens。query 同时使用事件顺序与实际时间差，不假设 event position 均匀采样，不引入 BPM adapter。

#### Query/content 的时点和层次

第一版选择共享 temporal 权重的 query/content 两种计算：

$$
c_j=C(H_j),\qquad q_i=Q(H_{i-1},t_i,e_i).
$$

选择理由是 post-commit content 可以复用，而 pre-row query 要随当前 gap 改变检索；两者的时点和输入易于检查。严格定义的 shifted-token 单流也可满足因果性，two-stream 不是防自见的唯一理论方案。这里只借用 [XLNet](https://proceedings.neurips.cc/paper_files/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf) 的职责分离，不采用 permutation LM；共享权重是本轮的参数共享选择。query 是短暂工作状态，没有第二套永久 K/V bank，也不接回旧静态 masked-block ActionReader。

记 $c_j^{(0)}$ 为该行已 commit 后的 local/relation 输入，$c_j^{(\ell)}$ 为第 $\ell$ 层 content 输出。第 $\ell$ 层历史条目保存的是 **layer input** $u_{j,\ell}=c_j^{(\ell-1)}$。预测 $i$ 时，$q_i^{(\ell)}$ 只读取事件 $j<i$ 的该层输入或其合法 coarse 摘要。构造 $c_i^{(\ell)}$ 时读取同一 pre-commit bank，加上自身的 $c_i^{(\ell-1)}$。不能改读“上一事件同层最终状态”而仍声称与此分层并行计算等价。

先由 `predict(State,t_i,e_i)` 得到分布，再由监督或采样分支处理该分布；随后 `commit(a_i)` 在私有新状态中验证完整 row、更新 exact/local/relation facts、构造本行各层 content，再执行归档与 eviction，最后原子发布供下一行使用的 state。query 和 content construction 使用各自明确的可见 IDs，不能共用一个含当前 target 的输入集合。给 joint head 的 $h_i$ 只取最终 pre-row query outputs。

#### 训练 carry 与推理 K/V

训练在声明的 TBPTT 边界保存 detached 的 layer-input states。在当前 chunk 内，以当前可训练的 normalization 和 K/V projections 读取过去：

$$
\bar u_{j,\ell}=\operatorname{SG}(u_{j,\ell}),\qquad
K_{j,\ell}=\operatorname{LN}_{\theta,\ell}(\bar u_{j,\ell})W_{K,\ell},\qquad
V_{j,\ell}=\operatorname{LN}_{\theta,\ell}(\bar u_{j,\ell})W_{V,\ell}.
$$

过去 writer 的生成图被截断，但当前 loss 对 normalization 与 $W_K,W_V$ 的读取梯度必须保留。缓存 `SG(LN(u) W_K)` / `SG(LN(u) W_V)` 会同时切断这部分投影梯度，即使固定参数下 forward 数值相同，也不符合训练合同。[Transformer-XL §3.2](https://aclanthology.org/P19-1285.pdf) 的类比用于这一“历史 hidden stop-gradient、当前读取仍可训练”的区别。

当前 chunk 内新提交的 content 保留图，后续预测可通过它更新 writer；不得每次 commit 都 detach。跨 chunk 时对所有需要保留的 Local/Relation/Temporal learned state 与 pinned descriptors 一并截断，精确事实继续保留。chunk 的 forward 信息连续，梯度范围遵循声明的 TBPTT 切点。

推理在固定参数、normalization 和缓存版本下保存已投影 K/V；同时保留 recent 的 layer-input states 供后续压缩，coarse layer inputs 也纳入快照与 parity 检查。投影缓存只属于 content bank。参数或缓存语义版本变化时失效重建，不跨 optimizer 版本复用。

#### 无空窗的 fine→coarse 转换

第一版取 recent 基数 $M=512$、压缩组长 $G_c=16$、coarse 容量 $S=64$。允许未压缩 recent 暂时增加到 $M+G_c-1=527$ 行，凑齐 16 条可归档行后一次性压缩并移出；所有 staging 行在转换前仍可读。

设预测前已有 $n$ 个 committed rows，定义

$$
c(n)=\left\lfloor\frac{\max(0,n-M)}{G_c}\right\rfloor,\qquad
R'(n)=\{G_c c(n)+1,\ldots,n\}.
$$

空历史时 $R'(0)=\varnothing$。coarse block $b$ 覆盖原始 row IDs $[(b-1)G_c+1,bG_c]$，在 $n=M+bG_c$ 的 commit 归档完成后可读；可见 blocks 为 $\max(1,c(n)-S+1),\ldots,c(n)$，不足一个 block 时集合为空，超出容量 FIFO 淘汰。

例如 n=513 时 row 1 仍在 recent；n=527 时 recent 有 527 行；n=528 时 rows 1–16 原子转换为第一个 coarse token，recent 为 rows 17–528。禁止让这些行先进入不可读 pending，再于后续行重新出现。逻辑顺序固定为 pre-row predict → 本行 content construction → archive/evict → 发布下一 state，不能在 content construction 前提前应用本行后的归档。

归档只由绝对 committed event 进度触发，padding、sample horizon 和 chunk 末尾都不是触发条件。dense chunk 的 bank 是逐 query / content 调用所需条目的有界并集：Q=128 内最多新生 8 个 coarse tokens，因此 coarse 临时候选可能达到 72；每个调用仍按自己的 birth/eviction mask 读取，而非整块套用 chunk 起点或终点的 bank。

#### 压缩对象与能力边界

第一版对每层**未经该层 normalization 的 layer inputs** 求固定均值：

$$
\bar u_{b,\ell}=\frac{1}{G_c}
\sum_{j=(b-1)G_c+1}^{bG_c}u_{j,\ell}.
$$

coarse token 保存 $\bar u_{b,\ell}$、起止 row IDs、起止时间、count 与 coarse-type。读 coarse 与读 fine 使用同一算子顺序：先取得 layer input 或其均值，再做当前 normalization 和 K/V projection，metadata 供相对时间/类型 attention bias 使用。训练中投影参与梯度，推理缓存该次投影。不能替换为“先 normalization 再平均”或“平均已投影 K/V”；这些计算一般不等价，推理归档仍须从保留的 raw layer inputs 构造。

固定均值是低分辨率历史 baseline，不加 compression reconstruction loss，也不扩展 learned-compression 搜索。一个 coarse token 不能提供 block 内逐事件的 query-dependent 选择：

$$
\sum_j\operatorname{softmax}_j(q^{\mathsf T}k_j)v_j
\not\equiv\operatorname{Attention}(q,\bar k,\bar v).
$$

上游 causal states 可能已经编码顺序，所以均值不一定完全无序；但它对前后组织和转换的保留没有充分性保证。M=512、Q=128 时，进入 coarse 的 writer states 已越过 TBPTT 边界，未来 coarse-read loss 不能直接回到当时 writer 教它如何适应压缩；当前读取参数仍获得梯度。[Compressive Transformer](https://arxiv.org/html/1911.05507) 支持分级压缩这一机制，不为本任务的均值摘要提供长程 pattern learning 保证。

#### Attention mask

SDPA 显式设置 `attn_mask` 与 `is_causal=False`，mask 根据绝对 event IDs、coarse birth/eviction、query/content 的自见规则与 padding 构造。PyTorch 2.11 的非方形 `is_causal=True` 使用 upper-left alignment，不自动理解前置 cached memory；布尔 mask 中 True 表示可读取。BOS 提供合法空历史入口，padding 不产生全 mask softmax 或伪造 EMPTY row。必须分别验证 query IDs、content IDs、forward parity 和梯度连接。

### 4.5 Joint head

首版采用共享 hand unary、完整 256-row 表和双手 coupling，不再展开 output-family 搜索。令 $h_L,h_R$ 是当前 pre-row query outputs、每手 action pair 有 16 类，coupling 维度 $r=16$：

$$
s(a_L,a_R)=u(h_L,a_L)+u(h_R,a_R)
+\frac{E_{a_L}^{\mathsf T}B(h_L,h_R)E_{a_R}}{\sqrt r},\qquad
B(h_L,h_R)=\tfrac12\left(W(h_L)+W(h_R)^{\mathsf T}\right).
$$

hand swap 转置 coupling，outer/inner 顺序保持。head 的输入类型和测试必须排除已经消费当前真值的 post-content；删去旧 GRUCell history path。legality 与 terminal mask 在 joint log-softmax 前生效，输出按四条 serialized lanes 转回完整 row。

可选 `model.clock_readout_hidden` 为每手 unary 增加一个直接物理时钟读出：
$u(h,a)+c(z,a)$。默认值 0 关闭该路径。$z$ 只包含 pre-row 的 lane LN age、
attack/release age、hand attack/release age、上一 event 间隔、已过去的曲长、
上一完整 row、occupancy、真实 terminal flag，以及已允许的未来时间 offsets/gaps。
共享 MLP 使用相同的 bounded/asinh 时间基和相对手坐标；它不读取 future actions、
源 LN 配对或 annotation，也不改变 joint-row 空间、coupling、合法性或 neural cache。

末层零初始化；从已有权重复制全部共享参数后，初始 logits 保持一致。
hidden=128、lookahead=16 时增加 152,080 个参数。
`training.clock_readout_learning_rate` 可指定独立 AdamW 学习率；各参数组共享
warmup、weight decay 和一次全局梯度裁剪。零末层使首个 update 先训练输出层，
随后梯度才能传入前层。该可选路径针对密集时间骨架下的快速重复按键问题，
其质量收益需由配对训练和完整生成检验，不能由参数可训练或零初始化一致性推断。

## 5. 资源合同：把“不随整曲长度 OOM”做成架构性质

### 5.1 起始资源配置

以下为 24 GiB Apple Silicon 上的起始配置。固定张量探针支持开始 M1–M2；完整模型执行包络需用实际训练/生成路径验证：

| 项目 | 起始值 / 硬边界 |
| --- | --- |
| hand hidden width | 128，左右手共享算子 |
| temporal blocks / heads | 2 / 4 |
| local layers | 3 |
| training microbatch | 2 |
| differentiable chunk Q | 128 source-event rows |
| recent memory M / staging | 基数 512；最多 15 行仍可读 staging，未压缩容量 527 |
| coarse group G_c | 16 rows；一次性压缩并移出 |
| coarse memory S | 64 tokens |
| relation neighbors K | 去重的完整 row 节点，每手普通节点≤64，加 active heads 后≤68 |
| dtype | 先保持 FP32；本轮不以 mixed precision 掩盖无界分配 |
| parsed-source cache | 同时限制最多 16 个、256 MiB owned buffers；CPU staging 单独计账 |
| MPS driver / process RSS guard | 首次完整模型 soft stop 4 GiB / 6 GiB；外层极限 8 GiB / 12 GiB，分别记账 |
| allocator ceiling | 8 GiB，以 8 GiB / MPS recommended maximum 设置 fraction |

令 $J=2$ 为 hands、$L=2$ 为 temporal layers、$d=128$、$H=4$、每元素 $s_b=4$ bytes。训练与推理的持久 temporal 存储分开核算，下面使用逻辑容量 $C=M+G_c-1+S=591$：

| 路径 | 持久 temporal 内容 | FP32、B=2 的主体容量 |
| --- | --- | ---: |
| 训练 carry | detached layer-input states；K/V 在当前 chunk 重新投影 | $BJL C d s_b$，约 2.31 MiB |
| 推理 layer inputs | recent 压缩所需 states 及 coarse inputs | 同上，约 2.31 MiB |
| 推理投影缓存 | 同一个 content bank 的 K 和 V | $2BJL C d s_b$，约 4.62 MiB |

推理上述主体合计约 6.93 MiB；训练还须计入当前 chunk 的投影、content/query activations 与 backward 保存。物理 ring 的额外 current slot、metadata、local buffers、relation payload/pins 另计，不把 query stream 再做一套永久缓存。

保守地令 chunk bank 上限

$$
T_Q=(M+G_c-1)+S+Q+\lceil Q/G_c\rceil+1,
$$

末项给 BOS。每种 stream 的一份 dense attention scores 为 $BJLHQ T_Qs_b$，起始配置约 11.38 MiB；两种 stream 的临时张量和 backward workspace 要分别计入。关系 K/V 按每 query 最多 68 个、每手宽度 128 gather 时约 34 MiB；若参数数量为 $P$，FP32 参数、梯度和 AdamW 两份 moments 约 $16P$ bytes，optimizer 临时 workspace 另计。这些是容量公式，不是最终训练峰值。

完整上界由固定参数、$O(BJLHQ T_Q+BJQKd)$ attention/activation storage、固定 buffers 和有界 CPU staging/cache 组成。整曲长度只能增加流式计算时间和落盘数据，不得增加设备 attention 维度、活跃计算图或常驻历史列表长度。

不把 FlashAttention/CUDA fused backend 当作 MPS 上成立的假设。普通 SDPA/math fallback 的实际 workspace 也必须符合预算。

#### 5.1.1 容量扩展与 Mac 训练配置

128 维、两层 temporal 的 1,281,820 参数配置保留为工程基线，不再作为模型容量上限。`oracle_time_train_mac.yaml` 将容量集中到跨行组织：facts/local/relation 保持 128 维，共享的 query/content 输入投影映射到 512 维，temporal 使用六层、八个 attention heads，joint head 读取 512 维输出。时间边偏置 MLP 单独使用 64 维，不随内容网络扩大。完整模型有 19,976,776 个参数，其中 temporal 占 19,006,512；local 515,072、relation 236,552、facts 78,848、head 139,792。coupling rank 16 已覆盖每手 16 种动作的完整矩阵秩，不为增加参数量扩大它。

该配置使用 FP32、microbatch 1、Q=64、effective batch 8。M/S、relation 的索引数与 LN pins 保持上述含义。Q=64 会缩短可反传的 writer 区间，不能称为与 Q=128 完全相同的训练。实际 prefix/target 仍保持完整；共享当前参数版本内的 prefix 可以节约重放。模型 width/layers、读写梯度和固定 loss 分母分别验证。

每个 update 先独立抽取两个 song-group/chart，再各抽四个 stratum/start/horizon，按 start 排序以复用前缀。这保留 batch-average risk 的期望，但窗口相关；日志中的路径概率是排序前单条 draw 的边缘概率，不是排序后位置密度。最多缓存两个当前版本的 prefix states，optimizer step 前清空，绝不跨更新复用。

Mac 配置采用四个 CPU threads；AdamW LR `3e-5`、weight decay `0.01`、clip norm 1，前 20 次更新线性 warmup。相同初始化与抽样序列的 200-update 对照中，`3e-5` 的固定 held-out NLL 为 2.8833，`1e-4` 为 3.0316；前者的早期完整生成也较少出现整体 LN 比例的大幅摆动。weight decay 的小规模对照未给出更改到 `0.001` 的充分证据。该选择仍须由更多训练和生成检验，不构成最优性或可玩性结论。

2,000 万参数的实测训练 checkpoint 约 229 MiB，采用 512 MiB 单文件硬上限。每 25 个完整 updates 及请求的最后一步发布一次，首次抽样前发布 update 0；中断可能回退最多 24 个已完成但未持久化的 updates，并从保存的 RNG/日志边界重放。这减少连续训练的写盘量，恢复不会拼接不同权重版本的状态。具体资源证据与失效案例见[实测报告](oracle_time_m3_validation.md)。

### 5.2 必须落实的保护

长 target 按 Q 分块，分块反传并释放图，跨 chunk 按第 4.4 节保留 detached layer inputs 和其余 learned carry。不能每 commit detach，也不能用 `retain_graph=True` 保存整段 sequence graph。detach 不改变 storage ownership：carry 必须放入自己拥有的固定容量存储，不能通过 view 持有整曲底层 tensor；也不能原地覆盖当前 backward 仍需的值。每行 logits、指标、生成文件和导出中间数据流式落盘。

同一次 optimizer update 内所有 sampled windows/chunks 保持参数版本不变，按固定 effective-batch 分母累积后再 clip/step。下一 update 用新参数重放 prefix。原始数组、精确事件索引与参数无关的 replay facts 可版本化复用；learned prefix states 不跨 optimizer 版本复用。

CPU 数据沿用 M0 owners，再加入磁盘数组、分块读取和双重 LRU 预算。M0 的完整 targets tuple 和 admission events 字典不是已满足该存储合同的证明。设备每次只接收有界 rows/features，不把完整 skeleton/targets `.to(device)`；首版不引入多 worker 各自复制的无预算 LRU。source byte admission 的起始上限为 64 MiB，超出显式报告支持包络，不静默删除高密度样本。

执行前做最大支持配置的 forward/backward 和长 rollout 实测，包含高密度、超长 LN、长静默间隔与长谱。请求超出已验证配置时提前拒绝或在运行前确定更小 microbatch；不要中途静默缩短语义 context、改变 M/S、删除候选 rows 或丢掉某些高密度样本。

OOM guard 不是“永不 OOM”的证明。目标是在声明的硬件/负载包络内保持有界并有余量；启动时同时记录系统 available memory、pressure 与 swap 增量，不能用物理总量代替当前余量。MPS fraction 乘的是 recommended maximum，不是机器物理内存；本机所测 8 GiB ceiling 对应 fraction 约 0.4504502，禁止用 0 关闭限制。

训练在 update boundary 预先保存 durable checkpoint；window 中途资源异常时显式终止，丢弃未完成的梯度累积，从最后 durable update 重放同一 draw，不恢复半个 autograd graph 或部分 optimizer update。不能指望 allocator OOM 后仍有资源新建 checkpoint。推理快照将 model/cache 版本、全部状态、RNG、next event ID 和已持久化输出位置共同落盘；恢复须校验并对齐文件边界，不能保留旧行又重新生成同一行。保存采用临时文件与原子发布，预留 staging 空间；失败不以 EMPTY、teacher forcing 或截短输出兜底。

生成的写盘间隔与资源检查分开：默认 `checkpoint_every_rows=512`，完成 prefill 和请求的生成段末也保存；导出直接使用已持久化的完成状态。中断恢复回退到最近一次成功发布。资源检查仍至少每 128 行一次，MPS 的持久 carry 在这些边界重新拥有存储并释放空闲分配，不依赖写盘来抑制后端保留。扩容模型的生成检查点实测约 45 MB，不能沿用小模型的高频写盘成本估计。

### 5.3 已有资源证据及适用范围

2026-09-15 在 Apple M5、24 GiB、macOS 26.6.2、Python 3.10.20、PyTorch 2.11.0 MPS/FP32 上，对原始 train/validation admission 与独立张量路径作过诊断。source parser/split 代码来自 `adfb1ee0dfaa7c0d35cdea03f015d11ef39ac71e`，设备 recommended maximum 为 19,069,665,280 bytes。结论限于下述 population 和探针，不宣称完整模型已验收。

Corpus 使用 14,689-row 4K index 与固定 annotation sources，按既有 song grouping、held-out 优先和 exact arrangement 去重得到 13,216 张 admitted 谱面：train 11,564、validation 1,652，合计 13,392,439 event rows；未读取 test source payload。index SHA-256 为 `2d2814c3b3ec47cd1247e8d50cef555e12ace7eae60a79944102926c142c2d8e`。观察到：

| 观测 | 数值 | 必需行为 |
| --- | ---: | --- |
| 最长谱面 | 26,976 rows，4,348.526 s | 流式 prefix、target 与输出 |
| 最大 16 s target | 532 rows | Q=128 分 5 chunks，不缩短 target |
| 最长 event gap / LN duration | 92.438 s / 35.375 s | gap 不 reset，open head 保留 |
| LN 最大 event-index 跨度 | 128 | 另测超过 recent/coarse 淘汰范围的合法 LN |
| release-only rows | 584,643 | 保留非 press 的合法 row |
| 30-note seed 不足 15 行 | 219 张，最少 10 行 | 必须实现短历史初始化 |
| admitted 但无合法 seed/target | 1 张 train，只有 24 hit objects | 显式任务资格过滤 |
| 最大 source 文件 | 12,438,026 bytes | 解析预算不能只按 note_count 估算 |

张量探针在 B=2、两手、d=128、两层/四头、FP32 下完成 72 次 forward/backward/update，Q 包含 1/16/64/127/128，分别使用强制 SDPA math 和默认 dispatch；含 relation gather、FFN 与 AdamW，占位 local 模块。9.48 s 内同步 phase 的 active/driver/RSS 最大值分别约 107.22/1,112.42/464.16 MiB。其 12,288 步固定容量 inference 的 active/driver 保持约 15.742/64.438 MiB。另将最长真实事件流单行送入固定 K/V ring，两遍共 53,952 行、36.94 s，active 约 5.058 MiB、driver 约 42.70–42.72 MiB，没有随消费行数增长。

这些探针没有完整 time-conditioned local、two-stream、修订后的归档/训练投影规则或 sampled generation，不能提供最终 peak、训练吞吐、长程组织质量或慢性泄漏的保证。active、driver、RSS 是重叠的计账视图，phase 采样也不保证捕获每个算子瞬时峰值。无需重复 census 或再做一轮通用算子 benchmark 才能开始 M1；完整路径实现后直接验证其执行包络。

## 6. training-window sampling 与序列 objective 共同定义

### 6.1 两种 sampling 必须分开命名

`WindowSamplingPolicy` 选择训练谱面、prefix 和 target horizon。
`DecodeSamplingPolicy` 从当前合法 joint-row distribution 选择实际输出。

两者有独立配置、seed、日志和实验身份，禁止把输出 temperature 与训练样本权重混为一谈。

### 6.2 training sampler 的起始分布

沿用防泄漏的 song-group split，先按 song group、再按 beatmap 选择；annotations 不改变某首谱面的 pretraining 权重。

group 均匀用于避免大谱包支配训练，可行 stratum 均匀用于保留 seed 后短历史位置的曝光。这是本轮选择的 population risk，不是数据自然给出的唯一公平分布。

对每张谱面找出第 30 个 note 所在的完整 seed row。按 target 起点之前额外可用的历史事件数分三个 context strata：种子之后 0–63、64–511、512 及以上。先在可行 strata 中等概率选择，再均匀选择对应的 target 起始 event。全部边界在训练前固定；不能根据 hidden target 的 loss/style/recurrence 选择起点。

target 采用实际时间 horizon，第一版使用 1s、4s、16s 三个配置化尺度等概率选择。target 为从所选起始事件起、落入该半开时间范围的全部 source-event rows；至少包含起始 row；跨真实谱面终点则截至真实终点并提供 terminal 合同。三个 horizons 在每个合格 start 都可选，不以剩余完整时长重新筛选。参数仅是起始尺度，不宣称是音乐语义边界。

先完成 seed/target 资格过滤，再构造可行 groups、beatmaps、strata 和 starts。令 $G$ 为可行 groups，$C_g$ 为 group g 的合格谱面，$S_c$ 为谱面 c 的可行 strata，$I_{c,s}$ 为其中 starts，则 draw-path 概率为

$$
q(g,c,s,i,h)=\frac{1}{|G|\,|C_g|\,|S_c|\,|I_{c,s}|\,3}.
$$

不同 horizons 可能裁切为同一 target，日志保留 draw path；若报告 target 的边际概率，必须对产生它的 paths 求和。短谱缺少 strata 时只在剩余可行 strata 中重分配，不进行未记账的拒绝重抽。

每个 target 前的真实完整 prefix 都需因果 prefill，至少含最初 30 notes。prefill 采用 no_grad，只构造后续需要的 exact/local/relation/content states，不计算无用的预测 head 或 query stream。优化应保持相同 content、memory 生命周期与参数版本；不能把完整 prefix 改成最近 30 notes 或最近 512 rows，因为 contextual states 可能携带更早历史，两种初始化一般不等价。

按第 5.3 节 population 对所有可行 starts 精确计算：平均 prefix 为 376.690 rows、平均 target 为 46.405 rows，数量比约 8.117；1/4/16s 平均 target 分别为 7.530/27.737/103.949 rows，16s 占全部监督 rows 约 74.67%。这是 row 曝光份额，不是墙钟成本比、梯度份额或已获得长程结构监督的证据。训练报告同时记录 prefill rows、target rows 和 wall time。

16 秒高密度 target 超过 Q 只增加 chunks，不缩短 target。记录整个抽样路径概率、prefix notes/rows/elapsed time、有效 recent/coarse coverage、target 时间与行数、terminal 标志、prefill 费用与实际监督事件总量。

### 6.2.1 可选的已知时间空档混合采样

`windows.gap_sampling_probability=p` 默认 0，保留上述分布和 RNG 序列。正值启用仅按 skeleton 时间选择的补充路径：依次均匀选择可行空档时长层、song group、chart、空档边界和起点。默认时长层为 `[2,8)`、`[8,32)`、`[32,+∞)` 秒，只索引完整 seed 后、有下一行的边界；`gap_context_rows=32` 限制起点距边界的行数，最长 horizon 必须实际包含该边界。完整 prefix、半开 horizon 和真实终点规则均保持。

基础路径的概率乘以 `1-p`；空档路径的概率为 `p/(可行时长层数 × groups × charts × gaps × starts)`，各级计数取决于此前选择。相同区间可由多个 horizon 或空档产生，报告边际概率时须合并。此配置明确改变训练风险，不作回到原分布的 importance correction。空档位置、监督动作和 annotation 均不进入预测输入，时间编码仍遵守第 2.2 节的有界合同。验证窗口独立固定，不随训练混合比例变动。

动机是一次 100-update 实测的 35,441 个 target rows 中，只有 18 个后续空档达到 2 秒，8 秒及以上为零。时间前视编码器在这份曝光下几乎没有学到长空档响应。`p=0.25` 是待验证的补充曝光配置；须检查固定验证退化、跨空档 LN、完整生成结构及真实采样曝光，不能仅凭增加样本就宣称质量改善。

### 6.3 新主 loss：完整 continuation code length，固定尺度归一化

对 sample w 定义

$$
S_w=-\sum_{i\in I_w}\log p_\theta(a_i^\star\mid H_{i-1}^\star,t_{\le \min(i+r,N)},e_i).
$$

一次 optimizer update 累积 $B_{\mathrm{eff}}$ 个 sampled windows，主风险采用

$$
\mathcal L_{\mathrm{seq}}=
\frac{1}{B_{\mathrm{eff}} Z}\sum_{w=1}^{B_{\mathrm{eff}}} S_w,
\qquad Z=128\ \text{作为固定数值参考尺度}.
$$

Z 不是每个窗口自己的行数，也不是当前 microbatch 的随机 token 数。因此长窗口拥有更多总训练权重，窗口内的一次错误不会因为所在窗口长而被额外除以该长度。这个取舍必须和 1/4/16s 的抽样预算一起承认；不能宣称同时实现了等窗口、等 row、等时间三个不相容的目标。

所有 chunk losses 按同一个 `B_eff*Z` 分母累积，不能“每 chunk 求均值后再平均”，最后一个短 chunk 也不能获得额外权重。microbatch 是并发容量，$B_{\mathrm{eff}}$ 是该次 update 的窗口总数；ragged batch 中部分窗口已结束时不能改除以 active batch，累积多个 microbatches 也不能漏掉其数量。完成全部 sampled windows 后，统一 clip 一次、optimizer step 一次。记录 unclipped gradient norm 与 clipping frequency，固定 Z 不消除长窗口的梯度尾部。

要求在参数、history state、memory policy 和 terminal 条件相同的情况下，sequence cost 对纯计算分块可加：

$$
S([a,c))=S([a,b))+S([b,c)\mid\operatorname{Replay}(H,A_{[a,b)})).
$$

相同 prefix 下改变 loss window 的划分不能改变该行预测。TBPTT 会改变跨 chunk 的梯度路径，不得把这种前向/计分一致性写成与全 BPTT 梯度完全相同。

对固定参数和谱面，若位置 i 始终使用同一真实 prefix、memory policy 与 terminal 条件，且 horizon 不进入模型，则 teacher-forced $\ell_i$ 不依赖覆盖它的窗口。于是

$$
\mathbb E_w\!\left[\sum_{i\in I_w}\ell_i\right]
=\sum_i\Pr(i\in I_w)\,\ell_i.
$$

在 corpus 上对谱面与位置共同求和即可。改变 horizon 与 sum weighting 首先改变各位置的曝光权重，去掉旧 equal-block mean-row 对长窗口单次错误的额外稀释；它不自动增加 motif 一致性约束。该恒等式描述 forward cost；使用 stop-gradient 后的训练梯度还由切点决定。

报告并区分三个尺度：

| 尺度 | 首版含义 | 不能推出的结论 |
| --- | --- | --- |
| 监督窗口 | 1/4/16s 内全部 target rows | 长窗口不自动提供全局 motif 标签 |
| 前向 memory | 512–527 fine rows、最多 64 个 16→1 coarse 摘要及 relation/exact state | 显式存储长度不等于历史信息充分性 |
| 梯度路径 | target 内 Q≤128 的 TBPTT chunks；prefix no_grad；旧 states 的当前读取参数可训练 | 532-row window 分五段不等于 532-row full BPTT |

### 6.4 context 少时不确定性更大：如何反映

不要求 short-context 与 long-context 获得相同 NLL，不将较少 context 下对唯一真值的偏差都当作实现错误。训练预算由事先选择的 context strata 定义；概率模型通过保留较宽分布表达不确定性。

禁止用本 batch loss 的倒数、按窗口拟合的 entropy 或动态“难度分数”自动减轻难例。评估按 context strata、target duration、rollout position 分开报告概率校准、code length/rate 与生成结构；需要时使用只由训练集拟合的简单合法 prefix prior 作相对预测基线，但它不作为启动实现的前置。

### 6.5 可关闭的 row-composition / state-transition marginal loss

本轮不引入 style reader 或 RL reward。实现 history-conditioned structural marginal loss，直接从同一个 joint-row 分布求和，不增加独立“质量判定器”。

第一版将 `f_r(a,H)` 固定为三组：press count、按 outer/inner 与左右手坐标定义的完整双手 press 配置、当前 occupancy 下的 lane-wise state transition。使用固定等权 $\eta_r=1/3$；它们主要强调 row composition 与局部状态转移，不称为整体序列结构 loss，也不等同于 Jack/Trill/Tech。某些 occupancy 下 transition 受 legality 强约束而很容易，需分组报告代价和梯度贡献。

令 $p_l=\mathbf1[a_l\in\{\mathrm{TAP},\mathrm{LN\_START}\}]$，$\omega_l^-,\omega_l^+$ 为提交前后 occupancy，则三组分类值为

$$
f_1=\sum_{l=1}^4p_l,\qquad
f_2=((p_1,p_2),(p_4,p_3)),\qquad
f_3=((\omega_l^-,\omega_l^+))_{l=1}^4.
$$

后续加入 same-lane return / intervening-lane return / alternation 前，先用 complete rows 与 lane-specific predecessor 明确定义关系。不得给 simultaneous chord 内的 notes 编造先后次序，也不通过 target 选择不可见邻居。

对真值对应的结构值 `y=f_r(a_i^*,H)`：

$$
P_{\theta,r}(y\mid H)=
\sum_{a\in\mathcal A_i:\,f_r(a,H)=y}p_\theta(a\mid H),
\quad
\ell_{r,i}=-\log P_{\theta,r}(f_r(a_i^\star,H)\mid H).
$$

合法候选最多 80 个，可在固定 256-row 表上精确枚举。实现使用合法 log probabilities 的分组 `logsumexp`，不能以 epsilon 掩盖非法真值或空 target group。总损失为

$$
\mathcal L=\frac{1}{B_{\mathrm{eff}}Z}\sum_w\left[S_w+
\lambda_{\mathrm{struct}}\sum_{i\in I_w}\sum_r\eta_r\ell_{r,i}\right].
$$

要求 $\lambda_{\mathrm{struct}}\ge0$、$\eta_r\ge0$ 且 $\sum_r\eta_r=1$，配置值均有限。辅助项来自同一个 row 标签的聚合，是损失侧重，不是新增独立监督。固定历史下，每项真实 marginal 的期望 CE 都在真实分布最小，和完整 row CE 的分布最优点一致；有限容量与优化会改变取舍，不能保证 calibration 或生成质量改善。

保留 `lambda_struct=0` 的可训练基线；**三组非零路径也必须在本轮实现并验证**，覆盖精确边际、有效梯度、分组记录与完整分母。只有配置字段或恒零 stub 不算交付。非零系数为显式配置，不自动广泛搜索，不以实现该项为由宣称已学到长程组织。它不假设“重复越少越好”“变化越平滑越好”；实际长段组织仍由 free-running 输出检验。

训练初版保持 teacher forcing。不要将 generated prefix 与未调整的原始 suffix 真值直接拼接计算 CE；生成改变 occupation 后，源真值甚至可能不再合法。scheduled sampling、反事实重标和 policy-level 训练均不在本轮默认范围内。

## 7. 输出必须是 SamplingPolicy 驱动的完整生成

输出头保留合法 joint-row 概率，不做四条 lane 的独立抽样。主要接口不是“返回 logits”，而是“给定 State 和当前时间，产生一个完整合法 row、更新后的 State 与可追溯的抽样记录”。

顺序固定：

1. 根据真实 prefix occupancy 构造合法 support，去掉 all-empty，并应用真实终点 closure 条件。
2. 得到原始模型 log probabilities，另行应用有版本的可选 history-only prior。
3. 应用 temperature 和 nucleus/top-p，在剩余合法 support 内重新归一化并采样。
4. 提交 row，更新所有 exact/learned state，进入下一个 skeleton 时间点。

一般形式：

$$
\pi(a\mid H)\propto\mathbf1[a\in\mathcal A_i]
\exp\{(s_\theta(a,H)+\beta b(a,H))/\tau\},
$$

再进行已声明的 nucleus 截断。当前默认 temperature=1、top_p=1、beta=0，保留原始模型的正概率 LN_CLOSE；同时评估 top_p=0.95。早期长谱运行中，0.95 确实持续删除过 LN_CLOSE 概率，尚无证据证明它改善生成质量。分别观察模型概率、策略裁切与最终 LN 持续/terminal closure；不能把不同随机 seed 的差异单独归因于 top-p。greedy 是调试对照，不能成为唯一 rollout。

验证 $\tau>0$、$0<top\_p\le1$ 及合法候选 scores 有限；cutoff 处保留全部同分候选，避免 row-ID tie-break 破坏截断 support 的镜像对称。模型分布等变不等于相同 RNG seed 下逐行输出镜像，后者还需要 categorical 排列及随机数的显式耦合。普通 checkpoint 恢复要求固定 runtime/配置下结果可复现。

允许以后接入从训练谱面拟合的 chord/return/LN 行为软 prior；接口须区分其贡献，但 beta=0 的接口本身不算已经加入实际谱面 bias，拟合 prior 也不是本轮主干的前置。开启时必须声明所偏好的具体属性及其观测来源，不能用没有定义的总“playability”惩罚代替。任何 prior 都只能看生成历史和当前时间；不能硬禁止 Jack、Trill、重复、爆发，也不能把未经标定的“人类极限”写成 chart legality。语言生成中的 repetition suppression 不直接迁移到谱面，重复和轮换可能正是目标组织。

记录 raw model log probability 与实际 decode-policy probability，不能混报二者为同一个 NLL。nucleus 或可选 soft filter 必须保证至少保留一个合法候选；数值异常不能以原谱真值、EMPTY 或静默修改 LN 来兜底。

free-running 阶段在 seed 之后不再读取真实 action，不重新贴回真值上下文。输出可以跨任意计算窗口持续生成，并保留 LN carry。

## 8. 真实谱面验收，而非只看汇总 NLL

### 8.1 正确性合同

必须覆盖：当前/未来 target 更换不影响更早 logits；future LN endpoint 不泄漏；训练 teacher forcing 与逐行 replay 一致；完整 row 同时更新；unknown/truncated/BOS/padding 区分；镜像 equivariance；prefix/skeleton 无未来 action side channel；窗口分块无前向语义变化；checkpoint 恢复；nonempty/legality/terminal closure；实际 output schema。

dense/step parity 在固定参数、eval mode、相同 memory 更新规则下比较。关系邻居、有效性与 exact state 要先严格一致，再允许规定的浮点输出容差。

以下两项独立于 logits parity，必须纳入 M1–M2 测试：

- **训练梯度合同：** no-grad prefill / TBPTT 之前的 layer inputs 无 writer 梯度，当前 chunk 的 normalization 与 K/V projections 仍从历史读取获得梯度；同一 chunk 内新提交 content 可从后续预测获得 writer 梯度。用至少两个不同历史条目及非退化 query 隔离历史读取，避免单个 key 的恒定 softmax 或当前 self 分支掩盖 projected-KV 被错误 detach。另检验 coarse 的 mean→normalization→projection 顺序、参数固定时训练重投影与推理缓存的 forward 一致。
- **归档转换合同：** 同一历史在 fine→coarse 转换时始终有声明的 fine 或 coarse 访问；Q=1/17/64/128 下分别核对 query 可见 IDs、content-construction IDs、archive births/FIFO eviction 及 cache payload。至少覆盖 n=512/513/527/528、满 coarse 后的 eviction，以及跨多个 TBPTT 边界的 active LN；不存在先消失再出现的不可读 pending。

loss 测试还须覆盖 ragged effective batch、长 target 短尾 chunk、完整累积后 clip/step、正 lambda 的三组 marginals，以及 head 只能消费 pre-row query。不能用 forward 相同推断 detach 位置相同，或用一个总 NLL 替代上述检查。

### 8.2 输出级交付

从真实 held-out song groups 中固定一组 continuation cases，覆盖稀疏/密集、长静默、反复 lane return、hand alternation、chord 变化、LN carry/release 与真实终点。使用实际 source 片段，而非仅 toy fixtures；可复用人工确认的典型片段作展示，但不将其重新变成 style supervision。

每例交付原始前缀与后缀、teacher-forced 逐行代价、至少多个随机 seed 下 top_p=1 与 0.95 的完整续写、greedy 对照、时间对齐渲染、exact state/抽样摘要和 terminal forced-closure 统计。生成 `.osu` 或可无损导出的 row 文件，并用独立 replay 检查。生成相对原谱的 lane 选择不同不自动是错误。

### 8.3 评估分层

概率质量：sequence NLL、nats/row、prefix-context strata、target duration、生成位置，以及相同 seed 条件下的 calibration/合法先验对照。

生成结构：press/chord 分布、lane/hand recurrence、短 motif 的持续/转换、LN duration/occupation、长段退化和多样性。比较真实后缀与多次生成的分布，并区分 raw model、decode policy 和 terminal legality 的贡献。末行关闭不能掩盖此前异常持有，不用一个“全谱完全匹配率”代替。

资源：prefill 与 decoding 分开计时，rows/s、每行延迟分布、峰值设备/RSS、随谱长增长的 live memory、cache 命中/淘汰范围。

完整模型的资源验收覆盖固定 shape 的冷/暖重复、不同长度排列、真实最长 continuation 的多次运行，以及 prefill/forward/backward/update/decode/checkpoint/reset/unload 各阶段。逐类记录持久 buffers 的精确容量和 A/D/RSS、系统压力、吞吐随重复次数的趋势。warm 后 A 漂移 1 MiB、D/RSS 漂移 64 MiB 可作为初始报警线；至少 10 分钟 soak 也只是观测窗口，不是“未超阈值就无泄漏”的判定。结合重复运行是否平台、持续斜率与 reset/unload 后状态解释：active 平台不代表整个进程有界，RSS 未立即下降也不自动证明泄漏。

旧 bidirectional masked-reconstruction 的 NLL 不能直接与新 causal task 排名。首次整体改版比较回答“新 baseline 是否可训练、能否实际续写、表现为何”，并不拆分归因每个模块。当前旧 query-matching 候选不作为需要再次战胜的门槛。

## 9. 里程碑与依赖

实现状态：M0 的因果数据、seed、逐行 exact replay 与终点规则，M1 的在线主干，以及 M2 的窗口采样与序列训练已实现，接口与验证范围见
[因果数据、在线主干与序列训练说明](oracle_time_continuation.md)。M1 包含逐行与分块 teacher forcing、训练重投影、显式 detach 和推理 K/V；M2 包含按 group/chart/stratum/start/horizon 抽样、固定 effective-batch 分母、三组精确 marginals、content-only prefill、分块 backward 与完整累积后的单次 clip/step。M3 已补齐实际 sampling、磁盘数组、双重 LRU 预算、资源 guard、持久检查点与流式导出；实测范围和参数选择见 [M3 验证报告](oracle_time_m3_validation.md)。M4 的 corpus training 与同条件质量比较尚待完成。

| Milestone | 实现内容 | 完成条件 | 不允许替代成交付 |
| --- | --- | --- | --- |
| M0：因果数据与 state 合同 | skeleton、30-note seed、ExactReplay、pre/post state、终点规则 | 真实谱面逐行 teacher replay 与 prefix 构造正确；未来动作隔离和有界时间前视 | 再写一份旧 run audit |
| M1：在线主干 | 三层 causal local summary、row-node relation frontier、query/content temporal、joint head | query/content 可见性和 batch/step 一致；fine→coarse 无空窗；提交输出更新后续 features | 只验证一个 relation score 反例 |
| M2：sampling 与 sequence training | WindowSamplingPolicy、固定 effective-batch sequence cost、三组可关 marginals、chunked backward/content-only prefill | 真实不同 history/时长可训练；历史读取投影与 chunk 内 writer 获得规定梯度；正 lambda 路径有效 | 旧等 block mean-row loss、只有配置字段的辅助项或只有 forward parity |
| M3：实际生成与资源封装 | 两种 top-p sampling、持续 rollout、LN carry、导出、双重 cache 预算、durable checkpoints | seed 后纯生成；长谱合法非空；terminal 行可追踪；状态/输出一致恢复；实际路径资源有界 | 只输出 logits、teacher-forced accuracy 或算子级内存平台 |
| M4：真实 corpus run 与可玩结构 | 全语料曝光、固定 held-out case outputs、人工 gold 参照与参数选择 | 可复现训练与完整续写；对完整动作、节奏、LN 配合和段落衔接作具体判断，呈现可玩的组织而非仅合法输出 | 用少量更新的 NLL、合法率或任意 pattern 计数代替生成质量 |

M0 是基础；M1 与 M2 的设计需要共同对齐，不要求先获得旧任务的显著收益；M3 的 sampling/资源语义在 M0 即确定，不能最后补丁式修 LN；M4 使用已完成的同一系统。这里的 milestone 是一条主任务，不是五个可以分别“无收益、停止采用”的微型研究项目。

后续 M3–M4 沿用 M1–M2 已实现的训练缓存与归档合同。无需新增通用审计、重复 corpus census 或先用小型 NLL pilot 决定是否实施；报警阈值随完整路径验证校准，不能成为等待“表示充分性已证明”才开始实现的条件。资源 soak、平均 NLL 和模块存在性都不能单独替代完整交付。

运行预算由真实吞吐与目标监督事件量共同设置，必须统计 prefill 成本，不能只用 update 数量表达训练暴露。软件 smoke 可以很短；性能 run 的解释必须与其训练量匹配。先得到一套完整新 baseline，再决定是否需要分解 ablation，不先展开 architecture×sampler×loss×optimizer 网格。

## 10. 工程迁移边界与结束标准

在现有 `research/oracle_time_continuation` 中继续实现主干与 runner。`data.py` 拥有 source/seed，`replay.py` 拥有 exact facts/legality，`engine.py` 拥有 query/commit；扩展这些 owner，不重复建立 ExactReplay 或另一套 corpus 身份系统。保留旧实验可复现，不用一串开关让双向 masked 与 causal continuation 共享隐含 assumptions。

磁盘数组与有界输入逐步接在现有 data owner 下；canonical output 使用流式完整-row JSONL，未闭合 LN 只保留四个活跃引用。若导出 `.osu` 需要按 head time 排序，使用有界落盘后处理，不能因最早 LN 尚未结束就在 RAM 累积所有后续 TAP。SQLite 只是可选的排序/导出实现，不是模型依赖，也不要求先构建通用数据库层才能实现 local/temporal 主干。

可复用：原始 `.osu` admission、四动作表、ExactReplay 基础、hand-role mapping、source split/identity、数值 time basis、部分 joint coupling、资源日志与渲染/导出工具。

必须重写或重新审核：prefix feature builder、pace/relations、三层卷积 gather、temporal 主干、旧 decoder GRU/ActionReader 接口、window sampler、loss aggregation、generation API、snapshot schema。

旧 checkpoint 不做 exact resume。需要迁移 embedding 等可兼容参数时，只允许 weights-only initialization，记录来源，optimizer/memory/sampler 使用新状态；首个 baseline 可直接从头训练，不将迁移变成前置。

结束标准是：代码能用一份完整配置重现实验，固定真实前缀可生成完整合法谱面；所有 action-dependent features 来自实际已提交历史；训练读取梯度、chunk 内 writer 梯度和无空窗归档通过独立检查；长时间运行的全部常驻状态与资源有界；sequence risk、两种 sampling 与 terminal closure 的作用清楚。工程结果可以记录失败，但不能据此把可玩结构验收标为完成。质量判断须展示完整动作与进入/退出上下文，对照当前人工 gold，说明可辨认的 motif、延续和转换，并保留退化反例；重复、Trill、Jack 或高密度本身不是错误。没有实际演奏测试时，结论限于谱面结构预审，不能代替玩家体验。也不能把有界历史存储直接当作学会利用历史的结论。

## 11. 防跑偏检查

每个 PR/阶段说明都回答一个问题：它怎样推进“给定时间骨架、仅凭已提交动作历史进行真实逐行续写”？无法回答的改动不进入本轮。

以下行为视为偏离本计划：重新接入 future action context；先按真值整窗建 relation graph 再 masking；种子之后偶尔塞回真值修复生成；把 batch/chunk 起点当作真实 BOS；在中间窗口关闭所有 LN；以长 target 费内存为由偷偷缩短语义窗口；把 hidden action 用于图选边；重新接入 style/action reader；以 generic repetition penalty 禁止谱面重复；用独立 toy 表达力测试或微小 aggregate NLL 差异代替完整续写交付。

## 参考依据与身份

[R1] Pulsefield-model，`docs/formulation/notation.md`，公开提交 `a7ab19f26cd8dd05480c8376060aaced1af4f2a7`：完整 rows、四动作合法性、LN closure、committed-prefix 合同。

[R2] 同提交，`src/pulsefield_model/research/source_action_modeling/{local_representation,representation,time_basis,model,actions}.py`：现有双向局部支持、pace、ActionReader/GRU 与 action schema。本文提出的 causal path 不是这些代码已经实现的功能。

[R3] 同提交，`src/pulsefield_model/configs/hydra/source_action_composition.yaml`：现有 MPS 环境与资源 guard 起点。

[R4] Dai et al. (2019), [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://aclanthology.org/P19-1285/)，§3.2：历史 layer states stop-gradient 后用当前参数投影；支持本计划的历史生成梯度与读取梯度之分，不移植论文效果结论。

[R5] PyTorch 2.11 官方文档：[SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)、[MPS memory fraction](https://docs.pytorch.org/docs/2.11/generated/torch.mps.set_per_process_memory_fraction.html)、[active bytes](https://docs.pytorch.org/docs/2.11/generated/torch.mps.current_allocated_memory.html)、[driver bytes](https://docs.pytorch.org/docs/2.11/generated/torch.mps.driver_allocated_memory.html)：非方形 causal mask、recommended working-set ceiling 与重叠内存计账。

[R6] Huszár (2015), [How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?](https://arxiv.org/abs/1511.05101)：scheduled sampling 不自动保留标准 likelihood 的统计目标；本任务另有 occupation 变化导致 source suffix 非法的独立理由。

[R7] Yang et al. (2019), [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://proceedings.neurips.cc/paper_files/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf)，§2.3：query/content 的信息角色分离；不证明本任务必须采用 two-stream，不采用 permutation LM。

[R8] Rae et al. (2020), [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)，§3：分级 memory、固定 pooling 与 learned compression 是不同选择；本计划只采用未归一化 layer inputs 的固定均值，不移植保序、质量或 writer 梯度方面的结论。

[R9] [M0 因果数据与回放说明](oracle_time_continuation.md)，实现提交 `991d2f987f8a112220f5beb76bcf2d8a0903ade3`。第 5.3 节给出本计划所用 corpus 与资源诊断的范围、计数、环境和解释限制。
