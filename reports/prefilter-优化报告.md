# 预筛优化报告

日期：2026-10-06。对比用的是同一批候选，共 1214 篇（RSS 1154 + 检索词窗口 139，合并去重）。旧版代码是 tag `pre-prefilter-20261006`（`81b433c`）。验证库是 `/tmp/arxiv_verify_data/arxiv_daily.db`。线上库 `/root/.arxiv_daily/arxiv_daily.db` 只做了只读查询，文件修改时间在完整流程前后没有变化。大模型走本机 CPA `http://127.0.0.1:8080/v1`。画像和 judge 用 grok-4.7，粗排用 grok-4.3，TLDR 用 grok-4.5。

## 改了什么

### 1. 英文排除词，按核心证据按比例扣分

画像提示从 `profile-v1` 升到 `profile-v2`。除了中文 `not_interested`，还输出 `negative_terms`（3–10 个具体英文短语）。`learning`、`model`、`robot`、`control`、`data` 这类泛词会被丢掉。缓存里的旧画像如果没有 `core_terms` / `broad_terms` / `negative_terms`，会重新生成；大模型不可用时继续用旧缓存。

排除词只做整句匹配，`visual slam` 不会因为摘要里有 `visual` 就命中。扣分不再要求正向命中为 0：原始扣分上限 4，再乘 `max(0, 1 - core_score/8)`。大约四个精确核心短语把扣分收到 0。反馈里标成「不相关」的标题仍会贡献负向词，同样滤掉泛词。

这次画像的英文排除词是：visual slam、autonomous driving、quadrotor control、soft robotics、surgical robot、swarm robotics、powered exoskeleton、underwater vehicle。

### 2. 作者按规范化全名加分

不再用姓氏集合。库里的作者收成 `名首字母|姓`（大小写、重音、连字符、`Last, First` 都会折掉）。PDF 里只有姓的条目不进表。同一作者在库里出现越多，权重越高：1 次 0.5、2 次 0.75、3 次及以上 1.0，再乘 `author_overlap_weight`（0.6）。一篇论文最多加 `3 * 0.6 = 1.8`，和原来的上限一样。

这次库里能抽出作者的是 45/104 篇（其余 PDF 元数据没有作者）。旧逻辑是 115 个姓。新逻辑是 208 个全名键。

### 3. 短语 BM25，核心词和泛词分档，泛词只在后面补位

BM25 改成整段短语。三个词及以上的短语如果整句没命中，只给相邻 bigram 一半分，两边都是泛词的 bigram 不计。多词短语不再把 `learning` / `control` / `model` 拆开加分。

画像多了 `core_terms`（权重 1）和 `broad_terms`（权重 0.25）。整句每个词都是泛词时，即使被写进 core，也强制当成 broad。旧画像没有这两档、且候选不少于 40 篇时，按文档频率把最高频的大约三分之一降成 broad。

核心证据包括整句、混合 bigram（如 `whole-body control`），以及短语里真正有区分度的单词（如 `humanoid`、`loco-manipulation`、`retargeting`、`parkour`）。`motion`、`tracking`、`priors`、`adversarial` 单独出现不算。

送进粗排的顺序是：

1. 全部核心命中排在前面，按原有混合分排序。
2. 若核心命中少于 `llm_prefilter_limit`（200），再用「命中了泛词、且没有触发排除词」的论文按文本分补位。文本分是关键词分加短语 BM25，不含新近度。
3. 名单总数不超过 `source.arxiv.broad_backfill_limit`（默认 100，写在 `config/default.yaml`）。泛词补位一律排在核心命中之后。
4. 整池既没有核心也没有合格泛词时，才退回按 BM25/新近度补位，避免某一天完全空转。

粗排模型便宜，多出来的泛词论文交给它过滤。

### 4. 去重改到截断之前

已推荐（业务日之前）和已在库中的论文，用 URL 别名和标题加摘要的内容键，在打分前剔除。截断之后 `_filter_seen_papers` 仍再滤一次。已读论文不再占名额。

### 门槛和类别偏好

准入看文本分，不把新近度算进 `>= 1.5`。新近度只参与核心名单的排序和 lookback。有核心证据才进核心段。

类别偏好的提示词去掉了单独的 `robot` / `learning` / `control`。保留 humanoid、locomotion、manipulation、quadruped、legged，以及 reinforcement/policy/gradient、dynamical/stability/controller。权重上限仍是 0.8，低于一条精确核心短语的 2 分。这份库算出来是 cs.RO 1.00、cs.LG 0.62、cs.SY 0.19、cs.AI 0.10、cs.CV 0.08、cs.CL 0.03，和现有收藏一致，没有再改系数。

## 同一候选池的预筛对比

抓取候选池 16.3 秒，进程内存峰值 193 MB。旧版打分 1.2 秒，新版 4.6 秒。旧版关键词是当天早上的 12 个 canonical terms 和 3 条中文 `not_interested`。新版用 `profile-v2` 的 8 个核心词和 6 个泛词。

线上 2026-10-06 已推荐的 10 篇，日期等于业务日，平时的 `date < 业务日` 不会去掉它们。这次对比和完整流程都把这 10 篇的 URL 和内容键放进 seen（只写副本，线上库只读），等价于它们已经推荐过。

| 指标 | 旧版 | 新版 |
| --- | ---: | ---: |
| 预筛输出 | 200 | 100 |
| 其中核心命中 | 不区分 | 38 |
| 其中泛词补位 | 不区分 | 62 |
| 两版都在名单里 | 79 | 79 |
| 作者加分触发（全池 1214） | 709 | 486 |
| 作者加分触发（名单内） | 151 / 200 | 43 / 100 |
| 负向扣分触发（全池） | 0 | 19 |
| 负向扣分触发（名单内） | 0 | 1 |
| 泛词单独命中进名单 | 99 | 62（全部在核心段之后） |
| 名单里已读/已入库 | 37 | 0 |
| 去重后送入粗排 | 163 | 100 |

重合 79 篇。相对旧 200 的比例是 39.5%，Jaccard 0.36。旧名单里另外 121 篇大多是拆词 BM25 或新近度抬上来的，例如 LLM 推理、循环语言模型、流水线 ROV。新名单的 62 篇泛词补位是 sim-to-real、reinforcement learning、MPC、imitation learning 的整句命中，而且没有踩中排除词；没有关键词的论文没有被拿来凑数。

旧版先截到 200 再去掉已读，37 篇占掉名额，粗排只剩 163。新版先跳过 39 篇已读（含线上这 10 篇），再取出 38 篇核心和 62 篇泛词，粗排收到 100，里面没有已读论文。

负向扣分在名单里只压到 1 篇核心论文：DASH（手术机器人，命中 surgical robot）。它没有进入泛词补位，也没有进最终 10 篇。强相关的人形论文没有因为排除词被拿掉。

作者加分在全池从 709 降到 486。姓氏匹配会让 Wang/Zhang/Chen/Liu/Kim/Lee/Park 几乎都加分；全名键还要求名首字母一致。名单里仍有 43 篇加分，主要是和库里同一批人形机器人作者重名，上限仍是 1.8。

## 测试

`python3 -m pytest tests -q`：127 项通过。覆盖负向词过滤、按比例扣分、排除短语不拆单词、全名键与大姓、短语 BM25、文档频率降权、泛词补位排在核心之后且不超过 `broad_backfill_limit`、触发排除词的泛词论文不补位、去重不占名额、旧画像缺字段时重建或在模型不可用时回退。

## 新版完整流程

在副本库上跑 2026-10-06 的完整流程。候选池命中缓存，没有再请求 arXiv，lookback 没有扩大窗口。粗排收到 100 篇（38 核心 + 62 泛词）。其中 38 篇核心有上次试跑留下的粗排缓存，这次实际请求了 62 篇。judge 池 24 篇，新请求 7 篇。TLDR 写了 10 篇。结果写入副本，没有写入线上库。

耗时 130 秒。进程内存峰值 191188 KB，约 187 MB。状态 `completed`。

线上当天 10 篇已经推荐过，所以被放进 seen。新版给出的是另外 10 篇，和线上这 10 篇没有交集。

线上当天 10 篇：

1. I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning
2. InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation
3. Transporting Unsecured Stacked Payloads with a Quadrupedal Robot via Multi-Objective Reinforcement Learning
4. Dataset-Free Compliant Humanoid Loco-Manipulation with Dynamic Online Posture
5. Exploiting Hierarchical Controller Structure in Contextual Parameter Learning for Humanoid Loco-Manipulation
6. Continual Humanoid Motion Learning
7. Echo in the Steps: Learning Perceptive Humanoid Parkour with Gated Memory
8. Humanoid Rickshaw Pulling: Whole-Body Locomotion under Coupled Wheeled Loads
9. EgoHumanoid-V2: Human-to-Humanoid Transfer of Coordinated Whole-Body Skills for Loco-Manipulation
10. CoDance: Learning Reactive and Compliant Human-Humanoid Interaction from Video

新版 10 篇（judge 相关性，以及预筛来源）：

1. InterEvolve: Test-Time Evolution of Reward Programs for Humanoid Loco-Manipulation（核心，5 分）
2. Filter-Aware Fine-Tuning for Safe Humanoid Whole-Body Tracking（核心，5 分）
3. KungfuAthleteBot: learning high-dynamic humanoid motion from video with unified robust recovery（核心，5 分）
4. Contact as a Decision Variable: Capability-Tradeoff Contact Selection for Legged Loco-Manipulation（核心，5 分）
5. Humanoid Loco-Manipulation With Discrete VLA Model（核心，5 分）
6. Humanoid Badminton: Learning Dynamic Racket Skills from Limited Human Motion Data（核心，5 分）
7. DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models（核心，5 分）
8. Adaptive Mean Flow for Responsive Closed-Loop Robot Control（泛词补位，3 分）
9. TUCO: Curating Simulation Demonstrations for Sim-to-Real Robot Policy Co-Training（泛词补位，3 分）
10. Task Inference Beyond Least Squares in Behavioral Foundation Models（核心，3 分）

相关性：前 7 篇都在人形或足式的全身控制、移动操作、运动跟踪、跑酷上，和这份库的主题一致，只是换掉了已经推荐过的那 10 篇。DAWN 是四足跑酷，靠核心词 `legged parkour` 进来，不是泛词补位。两篇泛词补位进了最终名单，但 judge 只给了 3 分：Adaptive Mean Flow 是通用闭环控制，TUCO 是通用 sim-to-real 示范筛选，都没有落到人形全身技能上。第 10 篇 Task Inference 被核心段留下，judge 也认为它只是行为基础模型的通用方法。粗排和 judge 把大部分泛词论文挡在了前 10 之外。

## 回滚

```bash
cd /root/arxiv_daily
git checkout pre-prefilter-20261006
screen -S arxiv -X stuff $'\003'
sleep 3
screen -S arxiv -X stuff 'arxiv-daily\n'
```

回滚只动 screen `arxiv`。确认 `5555` 重新在听。`config/custom.yaml` 不在 git 里，不受 checkout 影响。
