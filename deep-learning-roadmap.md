# Deep Learning Roadmap

> 从零到研究/工业前沿的系统学习路线。基于 Stanford、MIT、Berkeley、UCL/DeepMind 的研究生核心课程体系构建，经过逐门课程的前置依赖、内容覆盖、公开资源可用性验证。

---

## Roadmap Overview

```
阶段 0 ── 基础（已完成）
  ✅ MIT 18.06 Linear Algebra
  ✅ Andrew Ng ML Specialization (Coursera)
  ✅ Karpathy Neural Networks: Zero to Hero
                    │
                    ▼
阶段 1 ── 深度学习核心
  ┌─────────────────────────────────────────┐
  │  Stanford CS231n (Spring 2025 十周年版)   │
  │  Deep Learning for Computer Vision       │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
阶段 2 ── NLP + Transformer
  ┌─────────────────────────────────────────┐
  │  Stanford CS224n                         │
  │  NLP with Deep Learning                  │
  └──────┬──────────┬───────────────┬───────┘
         │          │               │
         ▼          ▼               ▼
阶段 3 ── 三条独立分支（平行，互不依赖，按兴趣选择）

   分支 A            分支 B              分支 C
   LLM 方向          生成模型方向         强化学习方向
      │                 │                   │
      ▼                 ▼                   ▼
  CME 295          ⚠ 概率论补充        David Silver RL
  Transformers &   (CS229概率笔记       (UCL/DeepMind)
  LLMs             或 CS228 部分)       + Sutton&Barto
      │                 │                   │
      ▼                 ▼                   ▼
  CS336            CS236                CS285
  Language         Deep Generative      Deep RL
  Modeling         Models               (Berkeley)
  from Scratch         │
                       或
                  Berkeley CS294-158
                  (更偏实践)
```

### 上下游依赖关系说明

| 关系 | 依赖类型 | 原因 |
|------|----------|------|
| 阶段 0 → CS231n | 充分前置 | Coursera ML 提供概念基础，Karpathy 提供手写梯度能力，LA 提供矩阵运算。三者合在一起覆盖 CS231n 全部前置要求 |
| CS231n → CS224n | 强依赖 | CS231n 教授的 backprop、优化器、正则化、BN 是 CS224n 默认已知的 DL 基础。不经 CS231n 直接学 CS224n 会重复"跟着做但不知道为什么"的问题 |
| CS224n → CME 295 | 顺序推荐 | CS224n 用 1 讲覆盖 Transformer；CME 295 用一整学期深入架构细节（MQA/GQA、RoPE、MoE、量化） |
| CME 295 → CS336 | 顺序推荐 | CME 295 提供 Transformer 理论深度；CS336 提供从零构建 LLM 的工程能力。Karpathy 的 "reproduce GPT-2" 视频是两者之间的自然桥梁 |
| CS231n → CS236 | 弱依赖 + 概率补充 | CS236 需要的 DL 基础来自 CS231n，但 CS236 本质是概率课，需要额外补充 KL 散度、变分推断、MLE 等 |
| CS224n 与 CS236 | **平行关系** | 互不依赖。CS224n 是 NLP 方向，CS236 是生成模型方向，前置都是 CS231n 级别的 DL 基础 |
| Silver RL → CS285 | 有断层 | Silver 提供 RL 理论基础（MDP、TD、策略梯度），但使用线性函数逼近。CS285 假设你能熟练训练深度神经网络。需要 CS231n 的 DL 实战能力作为桥梁 |
| CS224n → CS285 | 无依赖 | RL 分支不需要 NLP 知识，但实际上 CS285 2026 版新增了 LLM RL 内容 |

---

## 阶段 0：基础（已完成）

### ✅ MIT 18.06 Linear Algebra

- **讲师**：Gilbert Strang
- **级别**：本科
- **角色**：所有后续课程的数学基础

### ✅ Andrew Ng Machine Learning Specialization (Coursera)

- **级别**：低于本科（面向大众简化版）
- **提供了什么**：梯度下降、损失函数、正则化、基础神经网络的概念理解
- **没提供什么**：矩阵微积分、概率推导、SVM 数学、EM 算法、学习理论——这些是 Stanford CS229 原版覆盖而 Coursera 版跳过的内容

### ✅ Karpathy Neural Networks: Zero to Hero

- **级别**：研究生实践深度
- **课程目录**：
  1. Micrograd — 从零构建自动微分引擎（反向传播基础）
  2. Makemore Part 1 — Bigram 字符级语言模型
  3. Makemore Part 2 — MLP（复现 Bengio et al. 2003）
  4. Makemore Part 3 — Activations, Gradients, BatchNorm
  5. Makemore Part 4 — 成为 backprop 忍者（手写反向传播）
  6. Makemore Part 5 — 构建 WaveNet
  7. Let's build GPT — 从零实现，跟随 "Attention Is All You Need"
  8. Let's build the GPT Tokenizer — BPE 从零实现（2h13m）
  9. Let's reproduce GPT-2 (124M) — 从空文件到完整 GPT-2 训练（4h）
- **资源**：[YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | [GitHub](https://github.com/karpathy/nn-zero-to-hero)
- **关键作用**：补上了 Coursera ML 到 CS231n 之间最大的断层——手写梯度计算

---

## 阶段 1：深度学习核心

### Stanford CS231n (Spring 2025 十周年纪念版)

> Deep Learning for Computer Vision

**为什么是最好的 DL 入门课：**

- CS231n 由 **Andrej Karpathy** 和 **Fei-Fei Li** 在 Stanford 创建（~2015），是深度学习教育的开山之作
- 虽然以"计算机视觉"命名，实际覆盖**整个 DL 基础**：反向传播、优化器、正则化、BN、Dropout、CNN、RNN、Transformer、ViT、Diffusion、CLIP、DINO、生成模型、自监督学习
- CS231n 被全球 AI 课程引用为前置或参考，是事实上的 DL 基础标准
- **2025 春季十周年纪念版**已公开上传 YouTube，讲师阵容：
  - **Fei-Fei Li** — CS231n 创始人、ImageNet 创建者、Stanford HAI 院长
  - **Justin Johnson** — CS231n 联合创始人（从密歇根回归 Stanford）
  - **Ehsan Adeli** — Stanford 助理教授
  - **Zane Durante** — Stanford 研究员
  - Guest Lecturers: Jiajun Wu, Ranjay Krishna, Ruohan Gao, Yunzhu Li

**级别**：Stanford 200-level（研究生，高年级本科可选）

**前置要求**：

- Python + NumPy 熟练 ✅（Karpathy 课程提供）
- 微积分 + 线性代数 ✅
- 基础概率统计 ✅（Coursera ML 覆盖足够）
- CS229 是软性前置，非硬性要求 ✅（Coursera ML + Karpathy 足够替代）

**课程目录（Spring 2025 十周年版）：**

| 模块 | 主题 |
|------|------|
| **DL 基础** (Lec 1-4) | 计算机视觉概述；线性分类器 (kNN, Softmax)；正则化与优化 (SGD, Momentum, Adam, LR schedule)；神经网络与反向传播 |
| **视觉感知与理解** (Lec 5-12) | CNN (卷积, 池化)；CNN 架构 (BN, AlexNet, VGG, ResNet, 迁移学习)；RNN (LSTM, GRU, seq2seq)；**Attention 与 Transformers**；**Vision Transformers (ViT)**；目标检测与图像分割；视频理解；大规模分布式训练 |
| **生成与交互视觉智能** (Lec 13-16) | 自监督学习 (**CLIP, DINO**, 对比学习)；生成模型 I (VAE, GAN)；生成模型 II (**Diffusion Models**)；**Vision-Language Models** |
| **应用与前沿** (Lec 17-18) | 3D 视觉；机器人学习；以人为中心的 AI |

**Assignments（2025 版）：**

| 作业 | 内容 |
|------|------|
| **Assignment 1** | 基础运算与图像分类（NumPy 手写，无框架） |
| **Assignment 2** | CNN 与 PyTorch（BatchNorm、Dropout、卷积手写 + PyTorch 实战） |
| **Assignment 3** | **Transformers、CLIP、DINO、Diffusion Models**（图像描述、自监督学习、扩散模型） |

**学习资源：**

| 资源 | 链接 |
|------|------|
| **首选视频（2025 十周年版）** | [CS231n Spring 2025 (YouTube)](https://www.youtube.com/watch?v=2fq9wYslV0A&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) |
| 备选视频（Karpathy 主讲） | [CS231n 2016 (YouTube)](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) |
| 备选视频（Justin Johnson 密歇根版） | [EECS 498-007 Fall 2019 (YouTube)](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) |
| 课程笔记 | [cs231n.github.io](https://cs231n.github.io/) |
| Assignments | [cs231n.github.io/assignments2025](https://cs231n.github.io/assignments2025/) |
| 2025 Lecture Notes (社区) | [GitHub: cs231n-2025-notes](https://github.com/raimbekovm/cs231n-2025-notes) |
| 数学补充 | [Matrix Calculus You Need for DL (Parr & Howard)](https://explained.ai/matrix-calculus/) |

**预估时间**：6-8 周（每周 40h）

---

## 阶段 2：NLP + Transformer

### Stanford CS224n

> Natural Language Processing with Deep Learning

**为什么是最好的 NLP 课：**

- Stanford NLP 组是全球顶级 NLP 研究中心（Christopher Manning 是该领域泰斗）
- 2026 冬季版经历了**重大改版**：从传统 NLP 课转型为 NLP + LLM 深度课程，新增 RLHF、DPO、Agent、RAG、推理等当前最前沿内容
- 作业设计从理论推导到 PyTorch 实现完整闭环
- 默认期末项目是实现一个最小化 GPT-2

**级别**：Stanford 200-level（研究生）

**前置要求**：

- Python + PyTorch ✅（CS231n Assignment 2-3 提供充分训练）
- 微积分 + 线性代数 ✅
- 基础概率统计 ✅
- 基础 ML 知识 ✅（CS231n 远超要求）
- CS231n **不是**官方前置，但 Stanford 学生广泛反映"学过 CS231n 后 CS224n 会轻松很多"

**课程目录（Winter 2026 版 — 当前最新）：**

| 周次 | 主题 |
|------|------|
| 1 | NLP 历史；词向量 (Word2Vec) |
| 2 | 反向传播与神经网络基础；语言模型与 RNN |
| 3 | **Transformers**（必读论文："Attention Is All You Need"）；期末项目指导 |
| 4 | **预训练** (Scaling, Systems, Data)；**后训练** (RLHF, SFT, DPO) |
| 5 | **高效适配** (Prompting, PEFT/LoRA)；**Agents, Tool Use, RAG** |
| 6 | **基准测试与评估**；**推理 (Part 1)** |
| 7 | **推理 (Part 2)**；Guest Lecture: 分词与多语言 |
| 8 | Guest Lecture: 可解释性；社会影响 |
| 9 | Guest Lecture: 多模态；Guest Lecture (John Schulman) |
| 10 | NLP 2026 开放问题 |

**Assignments（Winter 2026 版）：**

| 作业 | 内容 | 占比 |
|------|------|------|
| Assignment 1 | 词向量入门 | 6% |
| Assignment 2 | 神经网络基础、张量导数计算、依存句法分析 | 14% |
| Assignment 3 | 自注意力与 Transformers | 14% |
| Assignment 4 | LLM 基准测试与评估 | 14% |
| **期末项目** | 实现最小化 GPT-2 + 3 个下游任务（或自选） | **49%** |

**学习资源：**

| 资源 | 链接 |
|------|------|
| **最新公开视频** | [CS224n Spring 2024 (YouTube)](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) |
| 课程主页（最新课纲） | [web.stanford.edu/class/cs224n](https://web.stanford.edu/class/cs224n/) |
| 2025 冬季存档 | [cs224n.1254 archive](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/) |

**与 CS231n 的衔接**：非常平滑。CS231n 教授的 backprop、优化器、正则化、BN 在 CS224n 中直接复用。CS224n 前 2 周会快速复习这些内容，有 CS231n 基础会感到"理所当然"。

**预估时间**：8-10 周（每周 40h）

---

## 阶段 3 分支 A：LLM 方向

### CME 295: Transformers & Large Language Models

> Stanford, Fall 2025 | Afshine & Shervine Amidi

**为什么需要这门课：**

CS224n 用 **1 讲** 覆盖 Transformer 架构。对于 LLM 方向，这远远不够。CME 295 用**一整个学期**深入 Transformer 的每一个组件和现代变体。

**级别**：Stanford 200-level（研究生）

**课程目录：**

| 讲次 | 主题 |
|------|------|
| 1 | Transformer 架构 (Tokenization, Embeddings, Attention, 完整架构) |
| 2 | Transformer 变体与技巧 (Attention 近似, MHA/MQA/GQA, RoPE, BERT 衍生) |
| 3 | 大语言模型 (MoE, 采样策略, Prompting, Chain-of-Thought) |
| 4 | LLM 训练 (预训练, 量化, 硬件优化, SFT, LoRA) |
| 5 | LLM 调优 (RLHF, Reward Modeling, PPO, DPO) |
| 6 | LLM 推理 (Reasoning Models, RL for Reasoning, GRPO) |
| 7 | Agentic LLMs (RAG, Function Calling, Agents, ReAct) |
| 8 | LLM 评估 |
| 9 | 当前趋势 |

**学习资源：**

| 资源 | 链接 |
|------|------|
| 课程主页 + Cheatsheet | [cme295.stanford.edu](https://cme295.stanford.edu/) |
| 视频 | [YouTube Playlist (Fall 2025)](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy) |

**与 CS224n 的衔接**：平滑。CS224n 提供 NLP 基础和 Transformer 概念入门，CME 295 在此基础上深入架构内部。

**预估时间**：4-5 周（每周 40h）

---

### Stanford CS336: Language Modeling from Scratch

> Spring 2026 | Percy Liang & Tatsunori Hashimoto

**为什么是最好的 LLM 工程课：**

- 由 Percy Liang（Stanford CRFM 创始人，HELM 评估框架作者）和 Tatsunori Hashimoto 主讲
- 这是目前唯一一门**从空文件开始构建完整 LLM** 的大学课程：tokenizer → 模型架构 → 分布式训练 → scaling laws → 数据处理 → 对齐
- 代码量"比其他课程多至少一个数量级"
- 取代了此前的 CS324（同一批讲师，CS324 侧重论文阅读，CS336 侧重实现）
- 被评价为"追赶 LLM 技术的最佳课程"

**级别**：Stanford 300-level（高级研究生——Stanford 课程体系最高档）

**前置要求**：

- Python + 软件工程能力（paramount）
- 扎实的 PyTorch
- 基础系统概念（内存层级）
- 微积分、线性代数、基础概率
- 深度学习实战经验推荐

**课程目录（Spring 2026）：**

| 讲次 | 主题 |
|------|------|
| 1 | 概述 + Tokenization |
| 2 | PyTorch (einops), 资源核算 (Roofline, Memory, Profiling) |
| 3 | 架构, 超参数 |
| 4 | Mixture of Experts |
| 5 | GPUs, TPUs |
| 6 | Kernels, Triton, XLA |
| 7-8 | 并行训练 (2 讲) |
| 9, 11 | Scaling Laws (2 讲) |
| 10 | 推理 (Inference) |
| 12 | 评估 (Evaluation) |
| 13-14 | 数据 (来源, 过滤, 混合, 重写, SFT) |
| 15-16 | 对齐 — RLHF, DPO, RL 算法 |
| 17 | 对齐 — RL 系统 |

**Assignments（5 个）：**

| 作业 | 内容 |
|------|------|
| **A1: Basics** | 从零构建 tokenizer、模型架构、优化器，训练最小语言模型 |
| **A2: Systems** | 性能 profiling、用 Triton 实现 FlashAttention2、构建分布式训练 |
| **A3: Scaling** | 理解 Transformer 各组件、拟合 scaling laws |
| **A4: Data** | 从原始 Common Crawl 处理为预训练数据（过滤 + 去重） |
| **A5: Alignment** | SFT + RL for 数学推理，可选 DPO 安全对齐 |

**学习资源：**

| 资源 | 链接 |
|------|------|
| 课程主页 | [cs336.stanford.edu](https://cs336.stanford.edu/) |
| 2025 春季存档 | [cs336.stanford.edu/spring2025](https://cs336.stanford.edu/spring2025/) |
| 视频 (2025) | [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy) |
| 代码 | [github.com/stanford-cs336](https://github.com/stanford-cs336) |
| 衔接桥梁 | Karpathy "Let's reproduce GPT-2 (124M)" 视频（阶段 0 已完成） |

**与 CME 295 的衔接**：CME 295 提供 Transformer 理论深度，CS336 将其转化为工程实现。有一定跨度（CS336 极度偏工程/系统），Karpathy 的 "reproduce GPT-2" 是两者之间的自然桥梁。

**预估时间**：8-10 周（每周 40h）

---

### 分支 A 完整时间线

```
CS224n (8-10 周) → CME 295 (4-5 周) → CS336 (8-10 周)
总计：约 20-25 周 ≈ 5-6 个月
```

---

## 阶段 3 分支 B：生成模型方向

### ⚠ 概率论补充（必要桥梁）

CS236 本质是概率论课程，数学强度**显著高于** CS231n 和 CS224n。需要先补充：

| 概念 | 说明 |
|------|------|
| KL 散度 | 衡量两个分布之间的差异 |
| 变分推断 | VAE 的数学基础 |
| 极大似然估计 (MLE) | 所有生成模型的统一框架 |
| 蒙特卡洛方法 | 近似积分 |
| ELBO | 变分下界 |

**推荐补充资源：**

- CS229 概率复习笔记（免费 PDF，精准覆盖所需内容）
- CS228 Probabilistic Graphical Models 部分内容（Ermon 也教这门课，概念共享）
- 预计补充时间：1-2 周

---

### Stanford CS236: Deep Generative Models

> Fall 2023 | Stefano Ermon

**为什么是最好的生成模型课：**

- 讲师 **Stefano Ermon** 是 score-based / diffusion 模型领域的**奠基人之一**（Song & Ermon 的 score-based 生成建模是该领域的开创性工作）
- 唯一一门从**统一的概率框架**出发，系统覆盖所有主流生成模型的大学课程
- 被 Stanford 学生评价为"在 Stanford 上过的最好的 ML 课"
- 不仅覆盖经典方法，还包含 2023 年前沿的离散数据扩散模型

**级别**：Stanford 200-level（研究生/高年级本科）

**前置要求**：

- CS229/221/228/230 任一（基础 ML） ✅（CS231n 超额覆盖）
- 基础概率与微积分 ⚠（需上述概率补充）
- Python 熟练 ✅

**课程目录（Fall 2023）：**

| 周次 | 主题 |
|------|------|
| 1 | 引言 |
| 2 | 背景知识 + 自回归模型 (Autoregressive Models) |
| 3 | 极大似然学习 + 变分自编码器 (VAE) |
| 4 | VAE（续） + 标准化流 (Normalizing Flows) |
| 5 | 标准化流（续） + 生成对抗网络 (GAN) |
| 6 | GAN（续） + 基于能量的模型 (EBM) |
| 7 | EBM（续） + **Score-Based 模型** |
| 8 | EBM（续） + 生成模型评估 |
| 10 | **Score-Based 扩散模型 (Diffusion)** + 离散潜变量模型 |
| 11 | **离散数据的扩散模型** |

**学习资源：**

| 资源 | 链接 |
|------|------|
| 课程主页 + 笔记 | [deepgenerativemodels.github.io](https://deepgenerativemodels.github.io/) |
| 课程笔记 | [deepgenerativemodels.github.io/notes](https://deepgenerativemodels.github.io/notes/) |
| 视频 (2023) | [YouTube (Stanford Online)](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8) |
| 课程大纲 + Slides | [deepgenerativemodels.github.io/syllabus.html](https://deepgenerativemodels.github.io/syllabus.html) |

**预估时间**：8-10 周（每周 40h）

---

### 替代选择：Berkeley CS294-158

> Deep Unsupervised Learning | Pieter Abbeel | Spring 2024

如果你偏好**更重实践**而非理论推导，这门课是 CS236 的替代：

- 覆盖生成模型 + 自监督学习（范围更广）
- 更注重实现，理论推导密度低于 CS236
- 2024 春季完整公开
- 课程主页：[sites.google.com/view/berkeley-cs294-158-sp24](https://sites.google.com/view/berkeley-cs294-158-sp24/home)

---

### 分支 B 完整时间线

```
概率补充 (1-2 周) → CS236 (8-10 周)
总计：约 9-12 周 ≈ 2.5-3 个月
（从阶段 1 完成后即可开始，不需要先完成 CS224n）
```

---

## 阶段 3 分支 C：强化学习方向

### David Silver RL Course + Sutton & Barto

> UCL / DeepMind, 2015

**为什么是最好的 RL 入门：**

- David Silver 是 **AlphaGo 的核心设计者**，DeepMind 首席研究科学家
- 10 讲把 RL 核心理论从零讲透，是全球引用最多的 RL 入门资源
- 配合 Sutton & Barto 教材（RL 领域"圣经"，第 2 版，免费 PDF）形成黄金组合
- Silver 讲座与教材 Chapters 1-13 有精确映射

**级别**：UCL 研究生

**前置要求**：

- 基础概率统计 ✅
- 线性代数 ✅
- 基础 ML 概念 ✅
- 不需要任何 RL 前置知识

**课程目录（10 讲）：**

| 讲次 | 主题 | 对应 Sutton & Barto |
|------|------|---------------------|
| 1 | 强化学习简介 | Ch 1 |
| 2 | 马尔可夫决策过程 (MDP) | Ch 3 |
| 3 | 动态规划 (Planning) | Ch 4 |
| 4 | 无模型预测 (MC, TD Learning) | Ch 5, 6 |
| 5 | 无模型控制 (SARSA, Q-Learning) | Ch 5, 6 |
| 6 | 值函数逼近 (Value Function Approximation) | Ch 9-11 |
| 7 | 策略梯度方法 (Policy Gradient + Actor-Critic) | Ch 13 |
| 8 | 学习与规划的整合 (Dyna, MCTS) | Ch 8 |
| 9 | 探索与利用 (Exploration vs Exploitation) | Ch 2 |
| 10 | 案例研究：经典游戏中的 RL | — |

**已知局限（2015 年课程）：**

不覆盖 PPO、TRPO、A3C、SAC、Offline RL、RLHF 等 2015 年之后的算法。这些由 CS285 补充。

**学习资源：**

| 资源 | 链接 |
|------|------|
| 视频 | [YouTube (David Silver)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) |
| 教材（免费） | [Sutton & Barto 2nd Ed (PDF)](http://incompleteideas.net/book/the-book-2nd.html) |
| 替代入门 | [University of Alberta RL Specialization (Coursera)](https://www.coursera.org/specializations/reinforcement-learning) — Sutton 本人参与，更结构化，4.8/5.0 评分 |

**预估时间**：4-5 周（每周 40h，含教材阅读）

---

### ⚠ 断层点：Silver → CS285

| 维度 | Silver 覆盖 | CS285 假设 | 差距 |
|------|------------|-----------|------|
| RL 理论 | MDP, TD, PG, Actor-Critic | 同上 + PPO, TRPO, GAE | 中等 |
| 函数逼近 | 线性特征 | 深度神经网络 | **大** |
| 框架能力 | 无 | 熟练使用 PyTorch 训练网络 | **大** |

**解决方案**：确保在 Silver 之前或同时已完成 CS231n（提供 DL 实战能力）。补充阅读 DQN、A3C、PPO 原始论文。

---

### UC Berkeley CS285: Deep Reinforcement Learning

> Spring 2026 | Sergey Levine

**为什么是最好的 Deep RL 课：**

- Sergey Levine 是 Deep RL + 机器人学习领域的顶级研究者
- 覆盖从 Imitation Learning 到 Offline RL 到 LLM RL 的完整 Deep RL 技术栈
- 2026 版新增 **LLM RL** 内容（RLHF 在 LLM 中的应用）
- 是唯一一门系统教授 Model-Based RL 和 Offline RL 的公开课程

**级别**：Berkeley 200-level（研究生）

**前置要求**：

- CS189/289A（ML 入门）或等价 ✅（CS231n 覆盖）
- RL 基础（MDP, 策略梯度） ✅（Silver 覆盖）
- 深度网络训练能力 ✅（CS231n assignments 覆盖）

**课程目录（Spring 2026）：**

| 讲次 | 主题 |
|------|------|
| 1 | 引言 |
| 2-3 | 行为克隆 (Behavioral Cloning) |
| 4 | RL 基础 |
| 5 | 策略梯度 (Policy Gradients) |
| 6 | Actor-Critic |
| 7 | 基于值的 RL (Value-Based RL) |
| 8 | 实践中的 Q-Learning |
| 9-10 | 高级策略梯度 (PPO, TRPO, Natural Gradient) |
| 11 | 变分推断 (Variational Inference) |
| 12 | RL 中的变分推断 |
| 13 | 控制即推断 (Control as Inference) |
| 14 | **LLM RL**（2026 新增） |
| 15-16 | Model-Based RL |
| 17-18 | Offline RL |
| 19 | 探索 (Exploration) |
| 20-23 | TBD |

**Assignments（5 个）：**

| 作业 | 内容 |
|------|------|
| HW1 | Imitation Learning |
| HW2 | Policy Gradients |
| HW3 | Q-Learning and Actor Critic |
| HW4 | LLM RL |
| HW5 | Offline RL |

**学习资源：**

| 资源 | 链接 |
|------|------|
| 课程主页 | [rail.eecs.berkeley.edu/deeprlcourse](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| 大纲 | [rail.eecs.berkeley.edu/deeprlcourse/syllabus](https://rail.eecs.berkeley.edu/deeprlcourse/syllabus/) |
| 视频 | [YouTube](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wHA9IChE26fC) |

**预估时间**：8-10 周（每周 40h）

---

### 分支 C 完整时间线

```
David Silver RL (4-5 周) → 补充阅读 PPO/DQN 论文 (1 周) → CS285 (8-10 周)
总计：约 13-16 周 ≈ 3.5-4 个月
前提：CS231n 已完成（提供 DL 实战能力）
```

---

## RLHF/GRPO 最小路径（不走完整 RL 分支）

如果你走分支 A（LLM 方向），只需要理解对齐技术中的 RL 部分，不需要完整的 RL 分支。

**最小要求：**

```
David Silver Lectures 1-7（必须到第 7 讲：策略梯度 + Actor-Critic）
   ↓
补充阅读（不需要完整课程）：
  - PPO 论文 (Schulman et al., 2017)
  - Yuge Shi: "A vision researcher's guide to PPO & GRPO"
    （专为非 RL 背景的人写的，从近乎零推导）
  - Nathan Lambert: RLHF Book（免费在线, rlhfbook.com）
  - Cameron Wolfe: GRPO 详解 (Substack)
```

**为什么是 Lecture 1-7 而不是 1-5：**

| RLHF/GRPO 需要的概念 | 来自 Silver 哪一讲 |
|----|-----|
| 策略作为参数化函数 | Lecture 7 |
| 策略梯度定理 / REINFORCE | Lecture 7 |
| 优势函数与 baseline（方差降低） | Lecture 6-7 |
| Actor-Critic 框架 | Lecture 7 |
| 值函数逼近 | Lecture 6 |

Lecture 1-5 提供 RL 通识但**不直接用于 RLHF**。Lecture 6-7 才是 RLHF 的理论根基。

Silver 课程之后仍需补充（Silver 不覆盖）：PPO 的 clipped surrogate objective、GAE、KL penalty、reward modeling pipeline、GRPO 的 group-relative baseline。

**预估时间**：3-4 周（每周 40h）

---

## 总时间预估汇总

> 基于每周投入 40 小时（全职学习强度）

| 路径 | 课程 | 时间 | 累计 |
|------|------|------|------|
| **主线** | CS231n | 6-8 周 | 6-8 周 |
| | CS224n | 8-10 周 | 14-18 周 |
| **+ 分支 A** (LLM) | CME 295 → CS336 | 12-15 周 | 26-33 周 |
| **+ 分支 B** (生成模型) | 概率补充 → CS236 | 9-12 周 | 23-30 周 |
| **+ 分支 C** (RL) | Silver → CS285 | 13-16 周 | 27-34 周 |
| **+ RLHF 最小路径** | Silver 1-7 + 论文 | 3-4 周 | 17-22 周 |

**典型组合时间：**

```
主线 + 分支 A（LLM 全栈）           ≈ 7-8 个月
主线 + 分支 A + RLHF 最小路径       ≈ 8-9 个月
主线 + 分支 B（生成模型）            ≈ 6-7 个月
主线 + 分支 C（完整 RL）             ≈ 7-8 个月
主线 + 全部三条分支                  ≈ 12-15 个月
```

---

## 课程级别总览

| 课程 | 编号体系 | 级别 |
|------|----------|------|
| MIT 18.06 | MIT 本科 | 本科 |
| Ng Coursera ML | — | 低于本科 |
| Karpathy Zero to Hero | — | 研究生实践深度 |
| EECS 498 / CS231n | Stanford 200-level | **研究生** |
| CS224n | Stanford 200-level | **研究生** |
| CME 295 | Stanford 200-level | **研究生** |
| CS336 | Stanford 300-level | **高级研究生** |
| CS236 | Stanford 200-level | **研究生** |
| David Silver RL | UCL 研究生 | **研究生** |
| CS285 | Berkeley 200-level | **研究生** |

这条路线覆盖的是**顶级 CS 硕士/博士项目 AI 方向的核心课程体系**。

---

## 补充资源

### 跟踪前沿（非系统课程）

| 资源 | 类型 | 说明 |
|------|------|------|
| **Stanford CS25: Transformers United** | 讲座系列 | 每周请一位顶级研究者做报告，任何人可通过 Zoom 旁听。用于跟踪最新研究，不用于系统学习。[web.stanford.edu/class/cs25](https://web.stanford.edu/class/cs25/) |
| **NYU Deep Learning (LeCun & Canziani)** | 完整课程 | 独特的 Energy-Based Model 视角。公开版本为 2021 年（DLSP21），部分内容已过时但 EBM 理论仍有价值。[atcold.github.io/NYU-DLSP21](https://atcold.github.io/NYU-DLSP21/) |

### 其他已验证的 LLM 课程

| 课程 | 学校 | 特点 |
|------|------|------|
| CMU 11-667 | CMU | 研究生 LLM 综述课，6 个作业，Slides 公开 |
| Princeton COS 597R | Princeton | Sanjeev Arora + Danqi Chen，研讨式，偏研究 |
| Berkeley CS294 LLM Agents | Berkeley | 专注 LLM Agent，MOOC 公开 |
