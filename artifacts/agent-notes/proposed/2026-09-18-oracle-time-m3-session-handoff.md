# Agent Note: Oracle-time M3 implementation and playable-generation session handoff

Note ID: 2026-09-18-oracle-time-m3-session-handoff
Status: proposed
Kind: process
Created: 2026-09-18
Updated: 2026-09-18
Product revision: f679269b92e96efb5bd7989e748bd42cf069379e; dirty product worktree on witness-style-probe, preserved by the file manifest below
Scope: Session handoff for M3 implementation, Mac training/inference, empirical quality, active experiments, and the next causal investigation
Related: 2026-09-17-oracle-time-mac-training-and-playable-continuation

## 接手目标与用户授权

最终目标是解决 Pulsefield 问题，让模型生成的谱面真正可玩；用户希望至少
优于 osuT5，并争取优于 Mug-Diffusion。这是目标，不是目前已经获得的结果。
本阶段仍然不输入音频，提供完整时间骨架，希望可玩结构已经能够涌现。
用户允许重新修订本阶段模型重建 plan，优化模型、训练采样、推理采样、
参数分配和工程实现，不要求机械地完成原有方案。

原始工作要求是实现
`docs/research/Pulsefield_oracle_time_causal_continuation_plan.md` 的 M3，真实跑通
训练、生成、导出和恢复的全路径，分析长时运行稳定性，保证 checkpoint
与写盘大小安全，并根据实测调整学习率和“遗忘”。用户已经明确：“遗忘”
指 **weight decay**，不需要再次确认。

用户多次授权增大必要模块的参数规模，并允许使用这台 Mac 的资源。最新目标
是 **param scale as need**；此前“param should not be too too big”的措辞已经被
更新。仍须选择在这台机器、现有 corpus 和 annotation dataset 上现实可行的
参数分配。当前 77M 是已经验证可承载的配置，不是用户规定的规模上限。

用户强调成果与问题本身，不追求短期任务完成。他建议从连接参数、激活、
信息传递、训练与推理 sample 等方向由现象追到机制，允许充分思考和实验。
不要为了交付一句“完成”而把合法导出、NLL 下降、LN 比例或模型变大当作
可玩性证明。最终决定质量足够时，必须用 `../beatmap-lens` 的判断尺度和
current human gold 例子检查机器生成的质量。

本次最后的直接请求是整理一份 handoff note，以便资源更充足的下个 session
接手。活跃目标没有完成，也没有被标记 complete。此笔记不是暂停所有已经
授权进程的指令；下面的有界训练和诊断在交接时仍可继续运行。

## 首先读取与工作边界

1. 在产品 worktree 读取 `AGENTS.md`、`README.md` 和上述 plan。实现接口说明在
   `docs/research/oracle_time_continuation.md`；实测报告在
   `docs/research/oracle_time_m3_validation.md`。
2. 从 `agent-notes` 分支读取本笔记和同目录的
   [研究 owner Note](2026-09-17-oracle-time-mac-training-and-playable-continuation.md)。
   owner 的本次诊断设计提交为
   `8d53ebd59699ea245ecdebe7018effa145ffb670`。本笔记是接手索引，
   实验记录继续追加到同一个研究 owner，避免分裂研究记录。
3. 产品改动全部仍在原 worktree，尚未提交或推送。不要 reset、clean、切走分支、
   删除未跟踪文件，或在只含 HEAD 的新 worktree 中误以为 M3 实现已经存在。
4. Notes 在专用 orphan `agent-notes` 分支；用 `git worktree list --porcelain`
   找到对应 worktree。不得把 notes 写入产品分支的 ignored note 目录。
5. 已经使用的技能：`hydra-conventions`、`pulsf-prose-standard`、`research-triage`、
   `pulsf-archive-agent-notes`、`pulsf-pre-push-checks`、全局
   `mania-pattern-judgment`。阅读 MPS 指南和
   `docs/guides/technical_analysis_writing.md` 时保留机制、证据和适用范围。
6. 研究 Note 保持 proposed，无 accepted revision/Experiment Card；当前研究评价为
   **REFINE**。用户单独明确授权了实现和实跑，不需要因为技能未自动授权执行而
   再次索要许可，但不能把探索性结果写成 accepted/SUPPORTED。
7. 没有授权产品或 note 的远程 push。没有创建自动化、其他用户 task 或新子代理。
   当前开发者指令禁止主动启动 subagent，除非用户或适用技能明确要求。

## 未提交代码的恢复材料

所有下面未注明前缀的实验路径，均相对于产品 worktree 内的
`artifacts/oracle-time-continuation/m3-20260917/`，下文简称实验根目录。
这是本任务明确使用的 artifact owner，不要广泛扫描其他生成 artifacts。

`session-handoff-20260918/` 保存本次交接的全部 50 个已修改/未跟踪产品文件：

| 文件 | SHA-256 |
| --- | --- |
| `manifest.json` | `06741fefea5ef0665660f1caec6e838672eda03633bcdcc3932d2aa27f1717fd` |
| `working-tree-files.tar.gz` | `2f9221fa33bda4520652932df2c4fed7d9176b69d241d63dd139381651a3c6a4` |
| `tracked-diff.patch` | `042e99062db727d0938b3ca9959f4871a3ec4cccc60f2f6f1ec27720669547a7` |
| `status.txt` | `bc3d7c1de8c22787d48fda91ba050837bf12ffd1717afa0bab9298e6efd9dd4b` |

压缩包 145,605 bytes，保存的是 `files/<repository-relative-path>`；manifest
逐文件记录 SHA 和大小。tracked diff 不包含未跟踪文件，所以不能单独依赖
patch 恢复。先核对当前 worktree；不要直接把快照覆盖到后续更新上。快照不含
checkpoint、dataset、其他 ignored artifacts，也不替代 clean product commit。

## 已实现的 M3 主体

代码 owner 是 `src/pulsefield_model/research/oracle_time_continuation/`，附近测试
在 `tests/research/oracle_time_continuation/`。现有 product diff 包括：

- 真实 corpus admission、源文件 SHA/split 校验、流式 canonical parser、磁盘
  row arrays、SourceStore LRU。每行 float64 time + 4 uint8 actions，共 12 bytes；
  LRU 最多 16 sources/256 MiB，staging 4,096 rows，source cap 64 MiB。
- M3 generation runner/Hydra CLI：完整 seed 后仅使用已生成历史和给定 skeleton，
  不读未来 actions/标签/LN 配对。joint-row legal sampling，CPU float64 policy
  arithmetic，tied nucleus，temperature 可配置；默认 temp1/top_p1/beta0。
- 全流程 `.osu` 导出、独立 replay/事件时间核对、继承展示用 BPM/SV/音频文件名，
  新 beatmap identity。源父目录没有音频，尚未制作附带音频的玩家测试包。
- 训练 checkpoint：完整 update 边界保存 AdamW、RNG、日志边界和参数；严格
  runtime identity，精确恢复。warm-start 只加载 weights、重建优化器，和 resume
  是不同操作。禁止跨 optimizer update 复用旧 learned carry。
- checkpoint compact CPU-owned tensors、实际写入硬上限、fsync/replace、每目标
  一个持锁 staging 目录。真实多次 SIGKILL 检查旧 checkpoint 保留、残留可回收。
  发布锁保护 publication，不代表整个 run journal 支持多个 writer。
- SQLite export 先设置 page cap，使用 `(time,lane)` WITHOUT ROWID 主键直接有序
  流式读出，不需要额外排序。header/note 写入前检查输出上限，失败保留旧文件。
- 长时 memory/available/swap/disk guards，MPS carry ownership refresh 与 checkpoint
  cadence 分离，卸载后不再额外保存。bounded/asinh time features 已替代早先危险的
  线性时间 age 输入。
- 完整前缀 replay、content-only prefill、批量 local/relation frontiers、temporal
  dense chunks、step/chunk/gradient parity；B8、cohort4 复用同一 update 的前缀。
- 可选 16-row 已知未来时间编码器：只提供有序时间 offsets/gaps，110,848 参数，
  shared-hand MLP、zero output init、独立 timing LR。plan 已明确修订最初“禁看
  未来时间”的限制，仍然严格 action-causal。
- Net2Net 风格 widening：只扩大 temporal 模块，复制 residual/FF/head channels，
  Q/K 缩放、输出列拆分并加有种子的零和扰动；清空旧 AdamW/cache。真实固定
  24 windows 上初始 NLL 仅变化约 5.3e-10。
- 可选 gap curriculum：默认概率 0 保持旧 RNG/采样；非零按时间 band/group/
  chart/gap/start 抽取，完整前缀、half-open horizon、只在真实末尾结束。记录实际
  population 和 mixture path probability；这是显式改变训练分布，没有偷偷做
  importance correction，也没有从源 actions 或 annotation 选 gap。

模块预算目前为 facts/local/relation 128 宽，temporal 6 层、512 或 1024 宽、
8 heads，time-bias MLP 64 宽，joint coupling rank16。20M/77M 参数精确值分别为
20,087,624 / 76,949,832（包括 timing）。不要把所有容量盲目加到 temporal；
前端瓶颈、训练目标和 conditioning 是否有用仍需证据。

一个显式 seed-count anchor 方案已跑负对照并从产品删除；冻结 v2 留存。
100 步没有表现出稳定的生成收益，不能无记录地重新加入相同方案。

## 机器、参数与安全边界

机器是 Apple M5 Mac17,3、24 GiB RAM、10 CPU cores（4P+6E）、macOS 27；
Torch 2.11 / Python 3.10 / psutil 7.2.2。训练用 CPU 四个 Torch threads，串行
rollout 用一个 thread。真实 ragged/full-prefix 工作负载上 CPU 目前更合算；
CPU/MPS 有并发负载差异，不应把数字当作纯设备 benchmark。

运行命令使用 `uv run --offline --extra mps`，测试额外 `--group dev`。系统
`python3` 缺少 psutil，使用 `.venv/bin/python` 或 uv。长任务用 `caffeinate -i`。
早期未 caffeinate 的 run 有较长 inactivity；性能报告必须区分 perf_counter
active time 与资源日志时间戳跨度。

当前主干 LR=3e-5，timing LR=1e-3，weight decay=.01，warmup20，structural
weight=.3，B8/cohort4/microbatch1/chunk64。这是探索中的工作配置，不是已经
优化完毕的结论。小片段测试 wd .01 与 .001 几乎无区别，尚无强证据降低 wd。
更新梯度经常被裁剪（约 98%），需要结合归一化、有效 target 数和各模块梯度
分析，不能只把 nominal LR 当作实际更新尺度。

20M profile checkpoint cap512 MiB、每25 update 保存；77M profile cap1 GiB、
每100 update 保存。77M AdamW checkpoint 约923.6 MB。77M guards 为
driver6 GiB/RSS8 GiB/min available2 GiB；MPS allocator outer cap8 GiB。
通用 output cap2 GiB 是每个文件的限制，不是 aggregate run quota；checkpoint
暂存时还需同盘容纳旧文件、新文件和 reserve。长期保留的 pinned stages 也计入
实际磁盘预算，不能无限创建实验。

2026-09-18 02:31 CST 附近检查：available 约10.6 GB，绝对 swap 约2.04 GB，
disk free 约236.9 GB。用 swap 增量判断本次负载；不要把早先已有 swap 当作本次
新增。所有这些都是快照，不是接手时的保证。

## 数据与验证身份

| 项目 | 身份或范围 |
| --- | --- |
| split | `artifacts/scoped-style-modeling/prepare-v1/split-manifest.json`，canonical SHA `15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a` |
| catalog | `artifacts/oracle-time-review/20260915-adfb1ee/catalog.json`，file SHA `e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28` |
| corpus | 13,216 sources；11,564 train/3,169 groups，1,652 validation/435 groups；11,563 eligible train |
| row cache | `artifacts/oracle-time-continuation/full-cache-v1`，约199 MiB |
| fixed ordinary validation | `mac20m-validation-manifest.json`，24 windows/1,126 rows；只覆盖很小的一组 validation charts，不是全 validation 估计 |
| current gold | `gold-validation-sources.json`、`current-gold-observations.json`、`current-high-feedback.json` |
| gap validation | `gap-validation-manifest.json`，17 boundaries，SHA `7be8253271318dc6fbd2bdde5bfce19e098094d351970cfaca7460a44ad9ee9a` |

没有使用 test payload。8 个 gold validation SHA 前缀为 5b69/e67f/ecc496/713ef9/
85058a/871955/98357f/ece738；完整值在 gold 文件。它们涉及 LN coordination、
纯 tap Stream、Jack、Trill、Tech、moving LN 等，不能只用三个例子下全局结论。

`../beatmap-lens` 保持只读，采用 canonical annotation reader 和 harness
render/inspection，不能直接把任意 JSON 都当 current gold。Foundation SHA：
`15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97`。
读取过的 current High 为171条，其中124 train、32 validation、15不在catalog。
不得改人类 annotation，不得把本 agent 的视觉判断叫作人类评审。

最长压力测试 source 是 **TRAIN**，SHA
`1022f1e408fcae61c784a7b0a390cd4b1f25a2720eaf86540207970c8c03b0ad`，
26,976 event rows、4,348.526秒（约72分钟）。源最大 LN 13.2秒、最长 gap84.735秒。
它用于压力测试与机制检查，不能证明 held-out generalization。

## 关键实测结果与失败证据

| 比较 | 固定24-window NLL / 主要现象 |
| --- | --- |
| 1.28M 基线500 updates | 2.68024257；44,580 targets |
| 20M LR1e-4/u200 | 3.03163328 |
| 20M LR3e-5/u200 | 2.88329591；相同74,354 targets的比较支持更小 LR |
| 20M LR1e-4/u500 | 2.75902986；183,849 targets，后续对照共用这个 init |
| seed-count anchor 100 added updates | control2.88268768 / anchor2.87705848；没有稳定生成增益，已移除 |
| 20M fresh AdamW control/time，added u100 | 2.56148032 / 2.55903806；相同35,441 targets |
| 同一 control/time，added u300 | 2.53359139 / 2.53678612；相同104,003 targets，未证明 timing 收益 |
| 77M widened/time，added u100 | 2.60371012；相同35,441 targets，比20M time-u100差约.0447 |
| timing-u100 再 gap25/u50 | 2.56367885；16,844 targets，尚未显示所需 gap 机制 |

以上不同起点、optimizer restart、采样及 capacity 的数字不可随意拼成单一
learning curve 或纯 size ablation。详见研究 owner 与原始 readout。

关键 pinned weights 均为各目录下 `weights.pt`：

| 目录 | SHA-256 |
| --- | --- |
| `quality-mac20m-u500` | `f3c991bc2c19f9ad2eb2bb3b4de6f93231a95a67d6fe8eaaebc896a398a59359` |
| `quality-skeleton-control-u100` | `3215d5026c9c95f5cdd9667f8ef1ca96aad1089955a63a121e93f2598313c9d8` |
| `quality-skeleton-timing-u100` | `5587eb3cff9ced0a46a71038a67b2a46816cac32625a355aa2f50da67675389e` |
| `quality-skeleton-control-u300` | `984d71a9f8ce9ca2138b9a40ed373ec67d14ff5e13a3b809b09fe2c14b32f387` |
| `quality-skeleton-timing-u300` | `008acf4f73c173b0f525e85664674a1188dd4917d263cc6d276ba044b76f550c` |
| `quality-skeleton-large-u100` | `b84ee573594ce65cc559753807441bb6f91172c31f0009664d3c2be22ee6aa25` |
| `quality-skeleton-gap25-u50` | `4edaf2363e6b4afafffce793495296111d20277f978dad2f38eeb411083e8c44` |
| `widened-time-init` | `cbfaccdfa2de4ac48e92eb9b613fb9bc3c611b47fe9757df9ae392d7cf742fca` |

原 timing-u100 的35,441训练 targets中，>=2秒gap仅18，>=4秒仅1，>=8秒为0。
gap25 增加了长 gap 曝光，但还没产生有用响应。17 held-out gap boundaries 上，
从 timing-u100 到 gap25-u50：pooled NLL2.53381068→2.52348162，boundary
NLL3.57156771→3.58127233，lane occupancy Brier .13425771→.14823265。
部分源例子有意把 LN 跨过8–10秒 gap，绝不能把“长空白前一律清空”作为真值。

固定生成历史，只把前方84.735/77.643秒 gap 压到125ms，timing-u100 的 any-hold
概率几乎不变；gap25-u50 仍是 .193355→.189723 与 .815346→.814062，变化很小，
实际长 gap 下反而略高。gap25 已停在首个50步 readout，不要盲目继续到100。

全长生成合法性与资源已跑通，但质量仍不合格。timing-u100 的最长谱生成
26,956行耗时1,483.618 active seconds，peak RSS492,994,560 bytes、swap增长0、
checkpoint45,524,359 bytes，却生成了 **89,356ms 的 LN**。这是明确反例。
新 indexed exporter 从同样完整 rows 导出32,692 notes，用时.6483秒，与先前
1,147,955-byte osu 逐字节一致；SHA
`27349bfed6576dec335c817b26f3533759235c14c26a8a64e562e574b9b1f476`。

同一 timing-u100 在 e67f 上 temp1/seed17 为660 LN/1,747notes；temp.85/seed17
为145/1,800，seed19为35/1,874。5b69/temp.85/seed17仍为1,305 LN/2,009notes。
低温不是通用 LN 开关，源 LN 比例也不是可玩性指标。完整 context 视觉复核见
`temperature085-context-review.json`：e67f两个seed均有 moving tap组织；5b69
有 staggered短LN与重叠，但源的对称重复和持续多键hold更强。尚无全谱可玩性
通过、玩家盲测或与竞争模型的匹配比较。默认温度仍为1。

77M/u100 三个完整 gold-reference输出5b69/e67f/ecc分别为3,368/2,341/2,683
notes，440/230/177 LNs，最长LN834/728/1357ms。必须继续检查多押密度和实际
组织；减少长LN同时增加大量多押，不自动等于改善。

## 刚完成与仍在跑的机制诊断

`mechanism-probe-v1/` 使用不可变 runtime v7，代码不修改模型参数。
`probe_common.py` 手工走 facts→timing→local→relation→temporal projection→
6 layers→head，并断言与正常 engine 输出一致。记录 RMS/delta、概率与干预。
诊断设计已先写入研究 owner `8d53ebd`；不是临时替换生产推理规则。

`branch_probe.py` **已完成**：e67f真实全前缀，在 zero-based256/512 行比较
source tap、首个tap改LN_START、tap列轮转三个合法分支；seed17/19/23/29，
temp1/.85，随后纯采样128行。48分支、6,144行，164.749秒。输出
`branch-readout.json` SHA
`de0e4e6b54300dc45a5107431af68df2fa3f8010629041861a90b740479afcd5`。

16次强制LN都在1–7行、61–425ms内关闭，没有censor。128行配对平均额外LN数：
temp1单LN分支+4.125，tap轮转控制+8.5；temp.85单LN分支−1，tap轮转0。
这不支持“一次LN必然触发持久LN吸引态”的解释。仍需看所有replicates和密度，
不能凭平均值证明没有任何状态反馈。该探针只有一个chart/两个位置。

立即下一行的新LN概率在256位置由.014132增到.107598、512位置由.050777增到
.210903。把tap分支hidden直接交给LN分支legal mask得到.087974/.167755；
完整LN分支与这个hybrid的JS仅.004633/.004265。大部分即时变化可以由support
重新归一化复现，但这个hybrid给另一历史使用了未必受训练的非法区域logits，
不能据此声称合法mask有bug、移除合法性或量化一个普遍“因果比例”。

`timing_probe.py` 在交接时**仍运行**。它比较 timing-u100/gap25-u50/timing-u300
三个权重，在最长TRAIN谱的source/generated两种固定历史、5969/10349位置，
实际/压缩future gap、仅当前timing移除、仅当前timing替换、仅最后16行历史
timing重算。每个case独立记录阶段激活、query-only timing梯度、relation
row-age/bias/attention描述。完整输出应为 `timing-readout.json`；若还不存在，
只把已有单case文件当partial，不能当全run结果。首两个case中timing差异RMS
仅约.000859/.000316、概率变化很小，但完整结果尚未解释。

两诊断都设30分钟active bound、CPU1、现有资源guard、总诊断artifact128MiB
上限；输出使用exclusive-create，不能向同一文件夹重跑覆盖。不要修改正在运行
的脚本或 `probe_common.py`；新诊断复制到新的版本目录再记录身份。

诊断代码的 SHA-256 为：`probe_common.py`
`d50ca79bc2fe52b493d83bbe9d3c5f49c20943c4417776bb16a9b26dc2c5583f`；
`branch_probe.py`
`84ed4be5c3bc12247cbfbbc890a19b9b462e360882de27da85f39a32d1ce6832`；
`timing_probe.py`
`80620454efb9c20d22bab75f62a8eb27098349efc28d64587274b52296f87b60`。

## 交接时进程与恢复

2026-09-18 02:39 CST 的观察（接手必须重新核对）：

| 工作 | 状态 / PID | 输出 |
| --- | --- | --- |
| 77M CPU continuation | Python PID86513，已计算236、81,773targets，继续至300；完整checkpoint已到200 | `skeleton-time-large/{updates,resources,windows}.jsonl` |
| timing mechanism probe | Python PID39393，已写9/12 cases，仍运行 | `mechanism-probe-v1/timing.log`、逐case JSON |
| branch mechanism probe | 已完成；原PID38784/session68740 | `mechanism-probe-v1/branch-readout.json` |
| 20M control/time pair | 两者300完成，104,003targets | 两个 `readout-300.json` |
| 77M/u100三全谱quality | 已完成 | `quality-skeleton-large-u100/readout.json` |
| gap25 training | 50完成，不继续 | `skeleton-time-gap25/readout-50.json` |

77M在02:31:33的checkpoint实际通过 `torch.load(...,weights_only=True,mmap=True)`
读出updates200、targets69,897、prefill593,684，文件923,599,593bytes。
`skeleton-time-large/weights.pt` 当时仍是100步；训练checkpoint与独立导出weights
不是同一个进度。不要拿它误报成200步weights。完成300时脚本会pin到
`quality-skeleton-large-u300/weights.pt` 并生成readout。

先从产品worktree运行：

```sh
ps -axo pid,ppid,etime,%cpu,rss,command | rg 'large_train.py|timing_probe.py|branch_probe.py'
tail -n 1 artifacts/oracle-time-continuation/m3-20260917/skeleton-time-large/updates.jsonl
tail -n 3 artifacts/oracle-time-continuation/m3-20260917/mechanism-probe-v1/timing.log
```

仅当77M进程已退出、尚未得到完整300步readout，且同一输出目录没有其他writer时，
才恢复；不要启动重复writer。命令必须使用冻结v4的原运行代码与配置：

```sh
PYTHONPATH=artifacts/oracle-time-continuation/m3-20260917/skeleton-time-runtime-v4/src caffeinate -i uv run --offline --extra mps python artifacts/oracle-time-continuation/m3-20260917/skeleton-time-runtime-v4/large_train.py large 300
```

**不要重新传 `large 100 300`**：100已经pin过，脚本会重复stage并撞existing目录。
若300 training已完成、仅pin或validation中断，先检查checkpoint/weights/readout
和pin目录，再只补缺失的只读validation，不要盲目重跑整个driver。

旧运行v3/v4使用较早publication实现；当前v7的持锁staging修复不会自动进入它们。
模型运行身份会hash所有module `.py`、Torch/Python/platform/device/threads。
不得编辑冻结snapshot或把当前源码塞进旧resume来绕过校验。想迁移当前runtime，
只能显式加载pinned weights开新的fresh-optimizer实验，不能假称精确resume。

## 冻结 runtime 与通用命令

| 目录 | `sha256.json` SHA / 用途 |
| --- | --- |
| `skeleton-time-runtime-v3` | `981dc3ef877c1a71bbf00b6116dcdd38022b1ef839704c2125dada9036375e27`；20M time pair |
| `skeleton-time-runtime-v4` | `c23912b30bcc47d3f3e08e689eadd338f02eb8297842a44b12a7279dbc58eb17`；large continuation |
| `skeleton-time-runtime-v5` | `d5e1697bcdf3cbb60560a54c2f44715a9f095baa1e0bdccc533f9d31a964f437`；publication修复 |
| `skeleton-time-runtime-v6` | `4bf72030c03591c97dab2709daee6736c82950950e28cd726720e441da411f80`；gap sampler |
| `skeleton-time-runtime-v7` | `efd670e285bef9a7987f8882e5baf09277549dbf334dc5883c9bfdab4b06a3a8`；indexed/page-capped export，最新 |

v7的 `quality_scaled.py` 接口是 `weights tag prefix-list p:seed-list temperature`，
按weights大小选大模型生成resource profile。旧quality脚本默认128MiB checkpoint
cap不能承载77M，不要因此误判模型失败。例如后续明确选择新输出目录后可用：

```sh
PYTHONPATH=artifacts/oracle-time-continuation/m3-20260917/skeleton-time-runtime-v7/src caffeinate -i uv run --offline --extra mps python artifacts/oracle-time-continuation/m3-20260917/skeleton-time-runtime-v7/quality_scaled.py artifacts/oracle-time-continuation/m3-20260917/quality-skeleton-large-u300/weights.pt quality-skeleton-large-u300 5b69,e67f,ecc4 1.0:17 1.0
```

上例需要先确认300步weights存在、hash并且输出目录尚未生成。不是当前已经
启动的任务。同目录的 `render_quality.py <tag>` 使用Lens harness生成完整
reviewContext分页和actions，必须看完整context，不能只看第一页。

## 已有验证及其准确范围

- v6 selected CPU suite：210 passed，4 deselected，21 subtests，98.05秒；日志
  `current-v6-cpu-checks.log`。命令为
  `uv run --offline --extra mps --group dev pytest tests/research/oracle_time_continuation tests/research/scoped_style_modeling/test_replay.py tests/test_package_layout.py -q -k 'not mps'`。
- v6 MPS：4 passed，189 deselected，76.14秒；日志`current-v6-mps-checks.log`。
- 此后只改indexed/page-capped export，当前runtime CPU owner：25 passed、2 MPS
  deselected，31.27秒；`export-indexed-checks.log`。没有声称重新跑过全部210项。
- gap/Hydra/runtime focused37 passed，2 deselected；default-off RNG、精确mixture
  mass、half-open/context/action隔离、cohort与update-resume均有针对验证。
- 实际CLI SIGKILL：e67f/timing-u100/temp.85/seed19，观测row640时杀子进程，
  正式Hydra resume后row journal和osu与不间断reference逐字节相同；三次子运行
  合计149.764秒，最终checkpoint44,706,567bytes。见
  `cli-recovery-temperature-{manifest,readout}.json`。
- 实际PyTorch writer内连续三次SIGKILL，以及active-writer staging保护测试；
  正常恢复留旧checkpoint，并回收一个目标最多一个stage的残留。
- 20M MPS generation recovery347.583秒、checkpoint约44.73MB、peak active/driver/
  RSS约518MB/1.218GB/828MB、swap增量0。此早期项是恢复output tail，非literal kill。
- 最大77M capacity只是每设备两次完整update，不能称长期稳定训练证明。
  1.28M MPS soak637秒/29updates稳定，也不能替代77M长期MPS证据。

接手若继续修改，先看最终diff，再选最小有意义测试；claim checks pass或push前
应用`pulsf-pre-push-checks`。本次新机制脚本只是artifact诊断，trace与legal
invariants在实际数据上运行检查，不加入镜像实现的无意义测试。

## 下一步的优先顺序与仍开放的问题

1. **先收集已运行结果。** 确认timing12case完整读出、77M300checkpoint/readout，
   分析资源增量；避免重复训练已完成阶段或覆写证据。
2. **解释timing信息为何弱。** 分辨MLP输出尺度/更新量不足、local融合衰减、
   temporal/head不使用该方向和训练目标本身不鼓励该信息。比较真实与生成历史，
   检查各模块grad/parameter update RMS、裁剪前后和zero-initialized路径；只看
   attention权重或激活范数不能证明因果。当前branch结果不支持强LN吸引态，
   别把这个假设当事实继续实现。
3. **检查训练/生成信息差异与有效采样。** TF总能看到真实最近风格，可能学会
   依赖近期真值而不保留seed结构；这是未验证假说。暴露数据、batch/cohort相关、
   horizon与loss归一化、rare events、temperature对序列分支选择都应结合实测。
   不要直接引入会造成非法target的scheduled sampling。
4. **relation row-age潜在外推。** `relation.py`仍有线性
   `(row_id-node.row_id)/32` feature；持续active heads或很久不动的lane可能使它
   大于训练范围。当前timing首两个sourcecase age仅169/173，bias幅度不到.6，
   未证明是病因。先测source/generated全分布、干预bias，再决定是否改表示。
5. **77M优化几何与模块容量。** widening拆分outgoing columns后，同一个AdamW
   nominal LR可能对应不同相对步长；这是未验证假说。没有新增temporal LR group。
   对比300步quality后再决定调LR、前端容量或规模。不是越大越好，也不应人为
   拒绝必要扩大。weight decay还缺辨识力强的比较。
6. **回到实际可玩结构。** 对值得保留的模型做全部8个gold context、多seed、
   不同chart类型的完整生成和Lens判断，检查移动、重复、双手组织、释放与攻击
   关系、过渡与长程结构，保留反例。LN比例/合法性/NLL只是辅助证据。
   尚未完成播放器实测，也没有与osuT5/Mug-Diffusion同输入同评价条件的比较。
7. **更新产品报告。** 当前 `oracle_time_m3_validation.md` 已写到gap baseline、
   temp.85 context、actual CLI kill、export修复；尚未纳入本笔记中的gap25负结果、
   20M pair300、77M100、branch probe及后续timing诊断。下一轮解释清楚后补到
   自足的产品分析，不把session流水账写进去。

保持真实结论：M3工程路径已具备并经多条真实数据运行；本阶段可玩结构与总体
目标仍未达到。不要标记goal complete，不要用一个局部好看的片段掩盖全谱反例。

## Next Lifecycle Condition

This handoff remains a proposed process record. The next session should append
new executed-run evidence to the related research owner, update product owners
for durable behavior, and retain exact experiment identities. Note acceptance,
implementation status, archival and remote publication require their separate
lifecycle authority; completing this handoff does not change the research goal
or mark the product implementation ready.
