# Agent Note: Oracle-time continuation 的资源核验与模块实现提案

Note ID: 2026-09-15-oracle-time-continuation-resource-and-module-review
Status: proposed
Kind: research
Created: 2026-09-15
Updated: 2026-09-15
Product revision: adfb1ee0dfaa7c0d35cdea03f015d11ef39ac71e
Scope: docs/research/Pulsefield_oracle_time_causal_continuation_plan.md 的公式、资源合同、抽样风险与待定模块；实际 train/validation corpus 和本机 MPS 的有界诊断
Related: 2026-09-14-source-action-relation-composition; 2026-09-15-source-action-relation-matching

## 结论与证据边界

计划的任务定义、V3 四动作合法性、固定尺度 sequence loss 和跨 chunk
stop-gradient 可以保留。`B=2, d=128, Q=128, M=512, S=64` 具备继续实现的资源依据。
现有证据不能把“完整训练与生成内存稳定”标为已验收：该提交只有计划，尚无完整
causal 模型可进行端到端验收。

实际核验覆盖：原始 source admission、整个可用 train/validation population
的谱长与抽样统计、六个真实极端谱面的独立 replay、全部 16 种 occupancy 的
support、CPU/MPS 的 cached SDPA mask 反例、分块计分与 memory ID 规则，
以及 MPS 张量工作区和最长真实事件流的固定容量存储。未训练新任务模型，未测试
生成质量，未把 retained masked-reconstruction 模型当作 V3 baseline。

建议先补齐下列实现合同，再按原 M0–M4 推进：

1. 区分预测前 query 与提交后 content；不能只凭 `is_causal=True` 排除 target。
2. recent/coarse 的 birth、visibility、eviction 由绝对 committed event 进度决定。
3. 内存预算计入两手、K/V、关系 gather、coarse 临时并集、参数/梯度/optimizer、
   CPU 数据与所有状态的 autograd 引用；不能只检查 attention score 大小。
4. 所有 sampled windows 使用同一个 effective-batch 分母；prefix 重放成本纳入预算。
5. 预先保存可恢复检查点。真实 allocator OOM 后不能保证还能分配内存完成新快照。

这是 proposed 实现方向，没有接受或采用状态；不替代原计划的完整系统交付。

## Repository State and Evidence

审阅对象为 `adfb1ee` 唯一新增的研究计划，generation contract 以
`docs/formulation/notation.md` 为准。可复用 source admission 和 split 的实际所有者为：

- `research/source_action_modeling/actions.py`：四动作 admission。
- `research/source_action_modeling/full_corpus.py`、`local_corpus.py`：原始 source
  验证、去重与 song-group 连通分组；以上路径均相对于 `src/pulsefield_model/`。
- `research/scoped_style_modeling/replay.py`：source parser；其 `prepare_chart`
  同时含 next attack、remaining duration 等双向事实，不能原样进入 causal features。
- `tests/research/source_action_modeling/test_full_corpus.py` 与 `test_state.py`：
  现有分组、source 身份、采样和快照测试的范围；它们没有证明新 continuation 正确。

产品 worktree 在核验开始时干净。核验期间另有未跟踪的
`src/pulsefield_model/research/oracle_time_continuation/` 和
`tests/research/oracle_time_continuation/` 出现；本诊断没有导入、修改或提交这些内容。
这些并行 M0 内容随后成为 `991d2f987f8a112220f5beb76bcf2d8a0903ade3`
（Implement M0 causal continuation data and exact replay）。其文档和接口已只读核对：
`data.py`、`replay.py`、`engine.py` 已拥有 source/seed/exact-state/query/commit；
后续实现应扩展这些 owner，不重复建立另一套 ExactReplay。
本 Note 的实测仍只归属于原计划及独立探针，没有为该并行提交签发测试或 review 结论。
被诊断脚本引用的原有 tracked source 在两个 OID 之间未改变。诊断脚本是本次生成的本地 artifact，
其 SHA-256 随结果保存，因此结果标为 exploratory，而非 accepted implementation evidence。

数据输入仅使用计划及现有配置指向的本地资产：

- `artifacts/indexes/beatmap_index_4k.parquet`，SHA-256
  `2d2814c3b3ec47cd1247e8d50cef555e12ace7eae60a79944102926c142c2d8e`。
- `artifacts/scoped-style-modeling/dataset-b22a7a4` 的固定 publication manifest。
- `artifacts/scoped-style-modeling/prepare-v1/split-manifest.json`。
- `dataset/` 中 index 明确定位的 `.osu` 和
  `artifacts/scoped-style-modeling/sources/` 中明确列出的 annotation sources。

未打开 test split 的 source payload。该结果不宣称验证整个磁盘上的所有谱面、
audio 内容去重或 index 未收录的资产。

## 1. 实际 corpus 的约束

按现有 admission、held-out 优先和 exact arrangement 去重规则，14,689 个
index rows 加上 publication sources 得到 13,216 张 admitted 谱面。
allocation 中另有 512 条 duplicate、58 条 rejected、1,448 条 heldout-no-read；
这些计数包含 annotation/index 两种来源，不能直接与 index 行数相加比较。

| 观测 | 数值 | 对实现的约束 |
| --- | ---: | --- |
| train / validation beatmaps | 11,564 / 1,652 | 保留既有 song-group split |
| train / validation song groups | 3,169 / 435 | group→beatmap 均匀抽样 |
| 全部 event rows / hit objects | 13,392,439 / 18,706,345 | note 和 row 必须分开计数 |
| release-only rows | 584,643 | 非空不能误写成必须有 press |
| rows p50 / p95 / p99 / max | 777 / 2,628 / 3,990.65 / 26,976 | 需要流式 prefix 和 target |
| 最大谱面跨度 | 4,348.526 s，约 72.48 min | 绝对时间及导出不能假设几分钟 |
| 最大 1 / 4 / 16 s 窗口 | 52 / 147 / 532 rows | 最密 16 s target 需要 5 个 Q=128 chunks |
| 最大相邻 event gap | 92.438 s | gap 不能当作 reset |
| 最大 LN 时长 | 35.375 s | exact open-head 不得保存未来 close |
| LN start→close 最大 event-index 差 | 128 | corpus 未覆盖 LN head 被 M=512 淘汰的情况 |
| 最小相邻 event gap | 1 ms | 不增加时间量化或最小 spacing |
| 30-note seed rows min / p50 / max | 10 / 25 / 59 | 30 notes 不保证 15 行局部支持 |
| seed 少于 15 rows 的谱面 | 219 | BOS/availability 是实际需求 |
| 达不到续写条件的 admitted 谱面 | 1 张 train，24 objects / 22 rows | 显式排除，不能降低 seed 门槛 |
| 单 source 文件最大字节数 | 12,438,026 | 解析峰值不能由 note_count 单独估计 |

58 条拒绝分别为：不足四个 source events 45、与 published split 冲突 6、
同 lane ambiguous overlap 4、零时长 LN 3。没有通过修时间或合并动作挽救非法 source。

六个极端 source 独立按事件建完整 rows，逐行检查 occupancy 与最终 closure，均成立。
其中最长谱、最高密度、最长 LN、最长 gap、最大 LN event 跨度、最短 seed 各有覆盖。
原始 source SHA、实际 rows 和 seed counts 保存在 `logic.json`。

**保留 active LN pin 的理由是生成合同，不是 corpus 中已经观测到 >512-row LN。**
模型可生成远长于训练样本的持有，尤其高密度行流可以迅速淘汰 start 的 neural token。
验收必须另加合法的 >512、>1,536-row LN 和多次 coarse eviction 的确定性边界输入。

### 抽样分布与成本

对每个可行 target start 用 `searchsorted(..., side="left")` 精确计算半开
1/4/16 s horizon，不以 Monte Carlo 估计。排除无 target 的一张谱面后，按
group→beatmap→可行 stratum→start 的概率加权，再等概率选择三个 horizons。

| horizon | 平均 target rows | target 超过 Q=128 的概率 | 平均 chunks | 总监督 rows 中的份额 |
| --- | ---: | ---: | ---: | ---: |
| 1 s | 7.530 | 0% | 1.000 | 5.41% |
| 4 s | 27.737 | 0.000549% | 1.000005 | 19.92% |
| 16 s | 103.949 | 28.622% | 1.300 | 74.67% |

平均 prefix 为 376.690 rows，平均 target 为 46.405 rows：prefix/target 数量比
8.117。它是事件数量比，不是已测得的墙钟成本比；prefill no_grad 和 backward
每行成本不同。按 update 数量制定训练预算会漏掉大部分 prefix 工作。
16 s 获得约四分之三的监督 row 暴露是已声明目标的结果，不需要暗中 importance weighting。

令 $g$ 为 group、$c$ 为 beatmap、$s$ 为可行 stratum、$i$ 为 start、$h$ 为 horizon：

$$
q(g,c,s,i,h)=
\frac{1}{|G|\,|C_g|\,|S_c|\,|I_{c,s}|\,3}.
$$

三个 horizons 在任意合法 start 都可选，并按真实终点裁切；没有“必须完整容纳
16 s 才可选”的额外条件。多个 horizon 可能得到相同 target，记录完整 draw path；
若需要该 target 的边际概率，应对产生它的 paths 求和。
先完成任务资格过滤，再构造 group 的可行 beatmaps/strata，不能无限重抽短谱而不记录概率。

## 2. 核心公式和约束审阅

### 2.1 因果分解、完整 row 和 terminal support

固定 skeleton 上的有限 categorical product 是合法归一化的条件模型。
限制每因子只看 $H_{i-1},t_{\le i},e_i$ 是该模型的在线信息约束；它不是对任意
conditioned-on-$\Gamma$ 数据分布都成立的独立性定理。
该任务直接观测真实 attack/release 时间并集，是 oracle-time baseline，其 NLL
不能代表未知 timing 或 audio-conditioned generation 的质量。

若 $o$ 条 lane 已打开，普通行 support 的**实际**大小为

$$
|\mathcal A(o)|=3^{4-o}2^o-1,
$$

对应 $o=0,1,2,3,4$ 为 80、53、35、23、15，而非每步都有 255 个合法候选。
使用固定 256 表并 mask 掉非法项仍是合理的实现，不必动态收缩张量 shape。

terminal support 大小为

$$
|\mathcal A_T(o)|=
\begin{cases}15,&o=0,\\2^{4-o},&o>0.\end{cases}
$$

因此任何合法 prefix 只要还有一个 skeleton row，就存在非空闭合续写。
这是枚举全部 occupancy 验证过的性质；不需要导出补隐式 release。
`is_terminal` 必须仅标记真实 skeleton 末行，并在 score 前施加 mask。

### 2.2 局部公式成立，但必须区分 commit 和 query

三层 dilation 1/2/4、三 taps 的 receptive field 为
$1+2(1+2+4)=15$，中间层分别为 3、7。原式的 $a_j$ 合法仅因为 $j$ 已提交；
不能把 $z_i(a_i)$ 放进预测 $a_i$ 的 residual、relation query 或 normalization。
在外层 attention 加 causal mask 无法修复这种直接泄漏。

有效事件数按实际原始 row 支持的并集计算；不要把重叠子摘要的 valid counts
相加成 $3^\ell$。LayerNorm 只沿通道；不引入跨时间 BatchNorm 或全 target 统计。
当前 query gap 单独调制 read query；不回写旧摘要的 pace 或 time features。

### 2.3 sequence objective 正确；补 effective batch 和梯度合同

固定 $Z=128$ 的 sum NLL 对选定 window distribution 定义了明确风险。
不同计算分块只要参数、forward state、mask 相同，标量 code length 必须可加。
独立数值检查在 Q=1/17/64/128 上给出同一个 257-row cost：2.108203125。
这只检查求和规则；没有冒充完整模型的 forward parity。

若一次 optimizer update 累积 $B_{\rm eff}$ 个 sampled windows：

$$
\mathcal L_{\rm update}=
\frac{\sum_{w=1}^{B_{\rm eff}}\sum_{c\subset w}\sum_{i\in c}\ell_i}
{B_{\rm eff}Z}.
$$

microbatch=2 只是并发量。ragged batch 中一条 window 提前结束后，另一条末尾
不能改除以 active batch=1；累积 A 个 microbatches 时也不能漏掉 A。
整个 update 结束才 clip 一次、step 一次、重放新参数版本的 prefix。
所有 chunk 边界上的 learned state 显式 detach；TBPTT 梯度取决于截断规则，
与 full BPTT 不等价。`Z=128` 只调梯度尺度，不会消除高密度/长窗口的梯度尾部；
记录 unclipped norm 与 clip fraction。

### 2.4 structural marginal loss 是重加权，没有新增监督

公式对合法 support 精确求和成立。实现用 masked `logsumexp` 求 target group
的 log mass；真值必须合法、group 至少含真值，不能靠 epsilon 修复空 group。
固定 $\eta_r\ge0,\sum_r\eta_r=1$、$\lambda\ge0$；非法负权重应在配置入口拒绝。

对固定历史，期望 marginal NLL 等于对应真实 marginal 的 entropy 加 KL。
因此与系数为正的完整 row CE 相加，在无限容量/可实现条件下仍由真实 row
分布共同最小化。有限模型和有限优化会改变取舍；不保证 calibration 或 rollout 质量更好。
该项来自同一个标签的聚合，不是第二个独立证据来源。

默认 `lambda_struct=0`。非零候选先只包含固定的 press count、双手 press 配置、
occupancy transition 三组；更复杂 return 事件先定义 simultaneous chord 的次序规则，
再接入，不能把同一 row 的两次攻击强行排成时间先后。

### 2.5 decode probability 与对称性

对合法 logit 或合法 log probability 加 prior，再除正 temperature、top-p 重归一化，
原采样式成立。验证 $\tau>0$、$0<p\le1$、所有有效 scores 有限，并记录 raw 与 policy
两套 probability。被 mask 的非法项允许是负无穷；有效项的 NaN/Inf 应显式失败。

共享手算子并不足以自动保证 nucleus 的镜像等变：按固定 row ID 打破边界同分，
可能使截断后的 support 不对称。建议保留 cutoff 处全部同分候选。
模型分布等变不意味着相同 RNG seed 会逐行生成镜像谱；同 seed 下的 categorical
排列也须有对应映射才有这种耦合保证。快照恢复验收固定同一执行配置和 runtime。

## 3. 待定模块的具体提案

使用 `research/oracle_time_continuation` 命名空间。以下定义后续预期职责；
已落地的 M0 `data.py`/`replay.py`/`engine.py` 保留其所有权，模块名字只作职责提示，
不声称并行提交已实现本节所有约束。

### Data / ExactReplay / export

- source storage 层（接在现有 `data.py` 下）：重用 admission 与 source/song identities，预处理为磁盘上的
  `times: float64[N]`、`actions: uint8[N,4]`，连同 source SHA 和 schema/version。
  时间差先在 CPU float64 计算，再变为网络 FP32；不把整曲 tensor `.to(mps)`。
- exact/learned state 层（接在现有 `replay.py`/`engine.py` 下）：四 lane occupancy、open start time/row ID、最近 attack/release、
  每手最近 attack 的完整 role mask；缺失用 availability，不用 0ms 冒充。
  `predict` 只读，`commit` 验证 next event ID 并原子提交完整四轨 row，拒绝重复 commit。
- 首版 parsed LRU 同时限制 16 entries 和 256 MiB owned buffers；超过 cache 单项预算
  的合法 source 使用磁盘数组分块读取。文件 byte guard 建议 64 MiB，超出显式报告
  预处理包络不支持，不静默丢弃。当前 admitted 最大文件约 11.86 MiB。
  解析时流式 hash/读取 HitObjects，避免全文 `splitlines` 和 Python 全量 lane-facts 表。
- 原始 parser 的对象 storage 估计：最大一张约 12.38 MiB，最大的 16 张合计约
  72.12 MiB；这不是解析峰值或 RSS。全文字符串、排序、tensor copies、worker prefetch
  还需单独预算。首版 `num_workers=0`，不设未计账的 per-worker LRU。
- canonical 输出逐行 JSONL 落盘。导出 `.osu` 时未闭合 LN 只有四个活跃引用；
  完成的 objects 可写 SQLite 临时表，再按 start/lane 排序流式输出。不能因为最早
  一个 LN 尚未关闭而在 RAM 缓存后续全部 TAP。中间 window 不触发 closure。

### Causal local summaries

三层共享左右手算子，各层使用可学习 offset-specific $W_{\ell,r}$，
actual elapsed time 和相对 hand/role actions 只生成逐通道调制：

$$
K_{\ell,r}(e)=\operatorname{diag}\!\left(1+\tanh g_{\ell,r}(e)\right)W_{\ell,r}.
$$

这使时间进入连接权重，避免为每条边生成 $d\times d$ 矩阵。
每层保留前 $2d_\ell$ 个输入摘要与对应 time/valid；物理 ring 可以多一个 current slot。
原始 row embedding、exact state 与 3/7/15-row 摘要分别有旁路。
首版 dropout=0，省去 prefill、step 和并行模式之间的随机 mask 歧义。
若加入 pace，使用最近 32 个已完成正 event gaps 的 running mean，并保留 count；
长 gap 仍是 gap，不能自动作为 rest。该数值选择是候选，不是已确认最优。

### Relation frontier

首版每 lane 保留最近 12 个 attack 和 4 个 release records，合计最多 64 个普通记录；
另 pin 最多 4 个 active LN heads。每个记录保存 bounded copied payload、完整 row
身份/动作、事件时间与相对 role，不持有整曲 tensor view 或无界前驱链。

每手一个 query，由两角色的 exact frontier 和局部摘要生成，读取同一 bounded bank，
用 own/other-hand、outer/inner-role 坐标表达关系。K=64 是普通记录总上限，
active heads 的最多四项单独记账。不能在“每关系类别 64”后再无界拼接。
同 row 可有多个不同关系，但相同语义键去重，以免重复存储偷偷加重 attention 权重。
已 committed 的 chord/close 才能形成关系；target lane 不参与邻居选择。

保留普通 dot-product content attention 和 additive relation/time bias，
不接入 deferred query-matching 实验的额外 $W_{rel}$。不同 query 的 standard
$q^Tk$ 不等于再次采用旧 relation-matching 因果干预。

### Temporal：query/content 分离与每行 visibility

推荐用同一套共享权重的两种调用明确时点：

- post-commit content $c_j$ 可以含 $a_j$，构造各层不可变 K/V。
- pre-commit query $q_i$ 来自 $H_{i-1}$ 的局部/关系摘要、exact state、当前 gap、
  terminal bit；只读 content $j<i$，永远不写成包含当前 target 的 cached content。
- content-stream 本层更新允许读自身及更早 content；query-stream 只能读严格更早。
  `commit` 生成 $c_i$ 并更新缓存。为避免重复预测的副作用，临时 query 计算结果不推进 state。

该分离借用 [XLNet 两种 attention stream 的防自见思想](https://proceedings.neurips.cc/paper_files/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf)，
此处固定时间顺序，不采用 permutation LM、双向上下文或论文性能结论。
单流 shifted-token 也可以合法，但必须另行证明其缓存对应 pre-action/post-action
哪个时点；在未给出该证明前，不推荐用一个含糊的 `h_i` 同时承担两者。

设预测前已有 $n=i-1$ 个 committed rows，$G_c=16$，

$$
R(n)=\{\max(1,n-M+1),\ldots,n\},\qquad
c(n)=\left\lfloor\frac{\max(0,n-M)}{G_c}\right\rfloor.
$$

完整 coarse block $b$ 覆盖 $[(b-1)G_c+1,bG_c]$，只在
$n\ge M+bG_c$ 可见；可见 block IDs 为
$\max(1,c(n)-S+1),\ldots,c(n)$。最近超限后先到期的最多 15 行位于 pending
accumulator，未形成完整 block 前不单独被 attention 读取。必须记录这个 visibility
规则，不能声称 recent+coarse 始终无缺口地覆盖最近 1,536 行。

pending 使用每层有限的 sum/count、start/end 元数据。完整块用 masked mean 压缩
各层待缓存的表示/KV，并附 count、时间跨度、event 距离及 coarse-type；不增加重建 loss。
active LN facts 仍独立精确保留。coarse 超过 64 时 FIFO 淘汰；绝不反写旧时刻的表示。
上下文表示可能已携带更早信息，1,536 是显式 rows 的存储规模，不是总信息影响的硬截断。

跨 segment stop-gradient 的类比来自
[Transformer-XL](https://aclanthology.org/P19-1285/)，recent+compressed storage 的近邻是
[Compressive Transformer](https://arxiv.org/abs/1911.05507)。此处采用确定性均值、
固定事件归档和无辅助重建目标，不能移植两篇论文的效果结论；这是适配组合，
没有已确立的新模型/目标创新性主张。

**并行 chunk 的 bank 不等于第一个 query 的 bank。** Q=128 期间最多新增 8 个
完整 coarse tokens，dense 实现可能需要 S+8=72 个 coarse 候选及逐 query 可见性。
独立 ID 检查覆盖 n=527/528 的首次归档和 n=1536/1552 的 FIFO eviction，并在
Q=1/17/64/128 保持相同可见 IDs；完整神经 forward parity 仍待实现验收。
也可先逐事件执行 reference engine，再优化为有限 bank 并集，不能按 chunk 末统一归档。

SDPA 用绝对 event ID、coarse birth/eviction、padding 构造显式 mask，
`is_causal=False`。布尔 mask 中 True 表示可见，见
[PyTorch 2.11 SDPA 文档](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)。
单 query、四个历史 values=0/1/2/3 的 CPU 和 MPS 反例中，直接
`is_causal=True` 得到 0；正确历史 mask 得到 1.5。非方形 causal mask 是左上对齐，
不会自动理解前置 memory。空历史用真实 BOS key，padding query 不执行全 mask softmax。

### Joint head / decode / checkpoint

每手共享 unary scorer，16 个 outer/inner action pairs 枚举成 256 行表。
沿用可镜像的 coupling 参数化，去除旧 GRUCell history：

$$
s(a_L,a_R)=u(h_L,a_L)+u(h_R,a_R)
+\frac{E_{a_L}^{\mathsf T}B(h_L,h_R)E_{a_R}}{\sqrt{r}},
\quad B(h_L,h_R)=\tfrac12\left(W(h_L)+W(h_R)^{\mathsf T}\right).
$$

取候选 coupling rank $r=16$；hand swap 会转置 B，serialized lanes 按 (1,2,3,4)
导出。legal/terminal mask 在 log-softmax 前生效。
默认 decode 为 temperature=1、top_p=0.95、beta=0，另输出 raw sampling 对照。

训练按固定 update boundary 保存 model、optimizer、sampler RNG 与配置/数据身份。
window 中途异常时丢弃未完成的梯度累积，回到最后 durable update 并重放相同 draw，
不把 half-applied optimizer 或部分 autograd graph 当作可恢复 checkpoint。
推理快照含 skeleton SHA、next event ID、全部 exact/learned/pending state、参数版本、
decode policy/RNG 和已持久化输出位置；采用临时文件+原子 rename，恢复时核对末条 row。

## 4. 内存预算与实测解释

记 $J=2$ 为 hands、$L=2$、$d=128$、$H=4$、每元素 $s_b=4$ bytes。
仅持久 temporal K/V 为

$$
C_{KV}=2BJL(M+S)d s_b=4.5\ \mathrm{MiB}.
$$

忽略 coarse 临时并集时，全部 temporal layers 一份 dense attention scores 为

$$
C_{score}=BJLHQ(M+S+Q)s_b=11\ \mathrm{MiB}.
$$

该数值不是 training peak。需要再计入 softmax/backward 保存、两种 stream、
coarse birth 并集、BOS、FFN、时间/关系 bias、投影与 activation copies。
每手每 query gather 64 个 relation K/V 的 storage 就约 32 MiB；pins 再加固定余量。
FP32 AdamW 的参数+梯度+两份 moments 约 $16P$ bytes，临时 optimizer workspace 另计。

硬上界应写为固定参数存储加
$O(BJLQ(M+S+Q)H+B J Q K d)$ 的 attention/activation storage、固定 buffers，
以及有界 CPU staging/cache。必须对 Local/Relation/Temporal、active heads 和 pending
accumulator **全部**做图截断；只 detach temporal memory 不够。
detach 的 view 还可能保留整块底层 storage；跨 chunk 保存到自己拥有的有限 tensor/ring，
训练期间避免原地覆盖 autograd 仍需的 values。禁止 `retain_graph=True` 贯穿整曲。

### 本机 MPS 测量

环境：Apple M5，24 GiB unified memory，macOS 26.6.2 (25G83)，Python 3.10.20，
PyTorch 2.11.0，torch source
`70d99e998b4955e0049d13a98d77ae1b14db1f45`，CPU threads=1，FP32，seed=17。
MPS recommended maximum 为 19,069,665,280 bytes，约 17.76 GiB。
环境中没有 `PYTORCH_MPS_*` 或 `PYTORCH_ENABLE_MPS_*` override。

所有 A/D/RSS 样本在 MPS synchronize 后取得：A 是 active allocator bytes，D 是
driver bytes，RSS 是进程 resident pages。A、D、RSS 互相重叠，不能相加为总物理内存；
表中 peak 是命名 phase 的采样最大值，不宣称捕捉到了每个算子内部的瞬时峰值。
[driver counter 包含 allocator pool 和 MPS/MPSGraph 分配](https://docs.pytorch.org/docs/2.11/generated/torch.mps.driver_allocated_memory.html)。

| 探针 | A | D | RSS | 解释 |
| --- | ---: | ---: | ---: | --- |
| Q≤128 的 72 个 F/B/update cycles，phase peak | 107.22 MiB | 1,112.42 MiB | 464.16 MiB | 包含 relation gather、两层 temporal、FFN、AdamW |
| 每个训练 step 释放 loss/grad 后 | 15.74 MiB | 最后 1,112.42 MiB | 最后 464.16 MiB | active 恒定；冷启动 shape/runtime cache 增长后留存 |
| cleanup 后固定容量 inference，12,288 rows | 15.742 MiB | 64.438 MiB | 约 466.1 MiB | 512-row 采样点 A/D 恒定 |
| 卸载模型并 GC/sync/empty_cache | 0 | 8.438 MiB | 466.09 MiB | RSS 没同步下降，不归因为 live tensor 泄漏 |

前 36 cycles 强制 SDPA MATH，后 36 使用默认 dispatch；Q 包括 1/16/64/127/128，
每组先固定 Q 再变 shape。不声称默认 dispatch 一定走某个 fused backend。
总计 9.48 s。这是算子级候选 workload：局部部分用 MLP，占位 relation，未实现完整
time-conditioned kernel、真实 coarse policy、双 stream 或生成采样。
不能由此给完整 baseline 参数量、训练 rows/s 或最终 peak 承诺。

另以最长真实谱面 SHA
`1022f1e408fcae61c784a7b0a390cd4b1f25a2720eaf86540207970c8c03b0ad`
的 26,976 rows 单行送入固定容量 MPS K/V，CPU 保留 source，设备每次仅接收一行。
两遍共 53,952 rows，36.94 s；A 恒为 5.058 MiB，D 约 42.70–42.72 MiB，
RSS peak 约 279.47 MiB。固定 ring 的长度和内存没有随已消费行数增加。
该路径读取真值作为已提交输入，不产生 sampled charts，也不验证 coarse semantics。

### 资源 guard 建议

1. 保留 `B=2,Q=128,M=512,S=64,d=128,FP32` 为候选起点。
   在完整模型验证前不扩大配置，也不把本次张量峰值当作部署上限。
2. 基于
   [MPS fraction 的 recommended working-set 定义](https://docs.pytorch.org/docs/2.11/generated/torch.mps.set_per_process_memory_fraction.html)，
   本机 8 GiB allocator ceiling 对应 `8 GiB / recommended_max = 0.4504502`，
   不是 `8/24`；禁止 fraction=0 关闭限制。
3. 初次完整模型 smoke 建议 soft stop D=4 GiB、RSS=6 GiB，allocator ceiling=8 GiB；
   原计划 8/12 GiB 可保留为外层极限，不能直接解释成安全 admission。
   只有完整模型实测后才冻结实际执行包络。
4. 启动时记录系统 available memory、pressure 与 swap 增量。一次本地快照 available
   约 8.00 GiB，开始时已存在约 4.82 GiB swap；不能只看 24 GiB 总量判定还有 8 GiB 空闲。
   已有 swap 本身不证明当前探针有泄漏。
5. 输出队列、指标容器、checkpoint staging、source workers 和 sampler coverage 都有限。
   设置 guard/预留 checkpoint 余量，并保存前一 durable checkpoint；真正 OOM 后仅做
   尽力清理和显式失败，恢复保证依赖已持久化 checkpoint。
6. `empty_cache()` 用于阶段诊断/卸载，不作为每 row 修复手段。fixed-shape warm soak
   要同时看 A、D、RSS、system pressure 和吞吐；只有 A 平台不足以验收。

## 5. 实现后的验收门槛

这部分是实施验收建议，没有签发 accepted Experiment Card 或启动模型训练的授权。

- M0：全部 admitted sources 做独立 replay；冻结 source/schema/split 签名，显式排除
  无 seed/target 的对象。对 target 改写保持已给定 prefix、当前时间与 terminal flag
  不变，验证更早预测不变；future LN endpoints 从模型输入类型中彻底隔离。
- M1：在真实 longest/densest/LN/gap/short-seed cases 上，用 Q=1/17/64/128 检查
  exact state、邻居 IDs、cache births/evictions 精确相同，eval FP32 logits
  初始容差建议 atol=1e-5、rtol=1e-4；若不达标先定位数值路径，不默认放宽。
  覆盖 n=512/528/1536/1552 和自造长 LN；逐步/批量 forward parity 是必做项。
- M2：参数版本固定，长窗口逐 chunk backward 后所有 carry tensors 都无 grad_fn；
  比较同一 detach schedule 的累计梯度。单独验证 ragged effective-batch denominator，
  不要求不同 TBPTT 截断规则的梯度相同。初始 lambda_struct=0。
- M3：完整模型至少覆盖 corpus 最长 26,976 rows 和更长合法边界流；多随机 seeds
  纯生成至 terminal，独立 replay/导出 round trip。异常恢复只能回到已持久化边界。
- 内存验收先在最大已支持形状 warm-up，再至少三个相同 shape cycle、三次最长完整
  continuation、不同长度排列及 no-grad prefill/teacher-forward/backward/update/
  decode/checkpoint/reset/unload 各阶段记录。建议 warm 后 fixed-shape A 漂移≤1 MiB、
  D/RSS 漂移≤64 MiB，至少 10 分钟同时检查吞吐与系统压力；这是一项待批准和执行的
  验收阈值，当前 9.48/36.94 秒诊断没有满足它。
- M4：validation 的 dense 16 s=532 rows、最长 10,565 rows、最长跨度 1,061.463 s、
  最长 LN=20.568 s、最大 gap=92.438 s 已可作为固定案例。记录多个随机生成及 greedy
  对照、完整 output 与 TF 概率分层。资源极值来自 train 不妨碍资源测试，但不能冒充
  held-out quality evidence。test split 保留到真正确定评估时再按授权处理。
- 训练预算以总 target rows、总 prefill rows、wall time 同时报出。只有完整模型真实
  吞吐后才能选训练时长；本探针不支持承诺具体 M4 收敛 update 数。

## Alternatives or Hypothesis Branches

| 分支 | 支持/反驳信号 | 当前提案 |
| --- | --- | --- |
| 两种明确时点的 query/content，共享 temporal 权重 | 无 target 自见且 step/dense 相同；额外计算使预算不可接受会反驳起始实现 | 优先，先 reference 后批量 |
| 单流 shifted action/time token | 需明确每个 cache 的时点、同样证明无自见；不能只提供总 NLL | 可行备选，当前未充分定义 |
| FIFO recent + 16→1 mean coarse | 生命周期有界、跨 chunk 相同；长期结构输出可能证明摘要过弱 | 默认实现候选，不宣称最优 |
| 更复杂 learned compression / 无限层级摘要 | 需独立的质量和成本证据 | 不加入首次 baseline |
| 关闭结构辅助项 vs 固定小权重 | 原始 row NLL/校准与 free-running 结构共同判断 | 首次默认关闭，保留明确接口 |

首版不通过延长实验、重新引入旧 reader 或 relation-matching 门槛来替代 M0–M4。
若分支差异后来需要因果比较，另外设计一张单变量 Experiment Card。

## Result Log: oracle-time-resource-review-20260915-adfb1ee

### Experiment and Reproduction

- Owning Note: 本 Note；Accepted revision: none；Experiment Card ID/revision: none。
- 用户于 2026-09-15 明确要求结合实际 corpus 和本机资源核验计划并提出模块实现。
  这授权上述有界诊断；不意味着采用提案或完整 baseline 已验收。
- Baseline source: `adfb1ee0dfaa7c0d35cdea03f015d11ef39ac71e`；模型 checkpoint: none。
  Intervention source OID: none；仅新建 artifact 探针，没有产品代码因果干预。
- Artifact owner: `artifacts/oracle-time-review/20260915-adfb1ee/`，新建目的地，
  JSON 输出使用 exclusive create，不 resume，不覆盖既有 run。
- 从 product root 执行，每条命令的 stdout/stderr 落在同一 artifact owner：

```sh
uv run --offline --extra mps python artifacts/oracle-time-review/20260915-adfb1ee/corpus_probe.py
uv run --offline --extra mps python artifacts/oracle-time-review/20260915-adfb1ee/logic_probe.py
uv run --offline --extra mps python artifacts/oracle-time-review/20260915-adfb1ee/mps_probe.py
uv run --offline --extra mps python artifacts/oracle-time-review/20260915-adfb1ee/real_stream_probe.py
```

重复执行需把脚本复制到新的 artifact 目录，先 census 产生依赖 JSON；原输出拒绝重写。
`corpus-summary.json` 记录 index/catalog/script hashes，`source-statistics.jsonl`
保存逐 source 的统计；`catalog.json` 和 `admission.json` 保存解析身份和 disposition。
catalog 签名为 `ea732530ae54de04c6f485b08bcce593f1bc5fc39a0557d5d30c3267149d269a`。
该签名沿用旧 SourceCatalog 的 max_source_rows/action-schema 描述，不是新 sampler 的身份。

`logic.json` 保存六个 replay、16 occupancy 枚举、SDPA 反例、memory ID 和 cost checks。
`mps-workspace.json` 保存完整 phase counters；`real-stream.json` 保存真实长谱两遍 counters。
所有 probe 源文件 SHA-256 都写入各自结果；raw artifact 不写入 agent-notes 分支。

### Results and Guards

- census 79.13 s，peak RSS 915.22 MiB；预设 guard 为 600 s / 2 GiB RSS。
- workspace 72 cycles + 12,288 inference steps，9.48 s；guard 240 s / D=4 GiB /
  RSS=6 GiB，allocator ceiling=8 GiB。
- real-stream 53,952 rows，36.94 s；guard 180 s / D=4 GiB / RSS=6 GiB。
- successful runs 均正常结束。没有网络数据下载、新模型 checkpoint 或 GPU 长训练。
- 第一版 workspace 探针在最终 cleanup 的 `del layer` 发生 NameError，未写出结果 JSON。
  修正该局部变量引用后完整重跑；原失败保留在 `mps-console.log`，成功记录为
  `mps-console-v2.log`。失败尝试不提供稳定性结论，产品代码未受影响。

### Plan Conformance / Evaluation

这是为计划审阅执行的 exploratory census/engineering diagnostics；没有 accepted
Card，也没有模型 baseline/intervention 对照。不存在可声称通过的质量主指标阈值。
工具脚本的修正和未完成完整模型，均已限定在证据解释中，不把工作区中的并行实现
纳入本次结果。

观测支持当前 corpus 可构造合法 causal 数据，并支持固定维度 kernel/cache 的
有界性；反驳“可直接使用非方形 SDPA causal flag”和“30 notes 一定给够 15 行”
这两个实现假设。内存理论不充分的主要位置是未计账状态/host storage 和生命周期，
而非这组初始 attention score 本身过大。

最强替代解释/局限：完整模型可能产生更多 autograd saved tensors、MPSGraph keys、
allocator fragmentation、host copies 或 checkpoint 峰值；短算子诊断无法排除这些。
现有 corpus LN 最长仅跨 128 event steps，不能证实 active-head 淘汰路径。

### Decision / Next Lifecycle Condition

Recommended outcome: **REFINE**。保留 causal continuation 主方向和起始配置，
用本 Note 的时点、mask、memory/sampler/resource 合同使待定实现可审阅。
不宣布 `SUPPORTED`，不更改相关旧 Note 的 lifecycle，不推送远端。

接受本 Note 需要人类明确接受其具体 note revision。产品实现与训练执行仍以相应
任务授权为准；完整模型达到上述 parity、真实生成、资源 soak 与恢复条件以后，
才能写端到端稳定性结论。相关旧 Notes 保持各自问题范围，没有被本 Note 整体取代。
