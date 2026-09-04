# OpenVLA 论文工程技术参考文档

> **论文**：OpenVLA: An Open\-Source Vision\-Language\-Action Model  
> 
> **arXiv**：[2406\.09246](https://arxiv.org/abs/2406.09246)  
> 
> **代码**：[openvla/openvla](https://github.com/openvla/openvla)（本仓库）  
> 
> **模型**：[openvla/openvla\-7b](https://huggingface.co/openvla/openvla-7b)
> 
> 



---



## 一、论文定位（读前必看）



OpenVLA 解决的核心问题是：**如何让机器人策略模型具备 Internet\-scale 视觉\-语言泛化能力，同时以开源、可微调的方式供社区使用**。现有 VLA（如 RT\-2\-X）虽性能强劲，但模型闭源、训练细节不透明，且缺乏在新机器人/新任务上高效微调的系统性实践，阻碍了广泛采用。



在世界模型领域，OpenVLA 处于 **「具身策略层」** 而非「世界模型层」：它不预测未来状态或视频帧，而是将当前观测 \+ 语言指令直接映射为机器人动作。但其训练范式（VLM backbone 迁移、动作 token 化、多 embodiment 数据混合）与大规模基座建设高度相关。



- **与团队目标「世界模型基座建立」的关联度：中**

    - OpenVLA 是 **策略模型（policy / VLA）**，不建模环境动力学或未来观测

    - 可借鉴之处：动作离散化与 LLM 统一、双视觉编码器融合、RLDS 数据管线、LoRA/量化微调、跨 embodiment 数据混合权重

    - 若团队基座路线是「VLM \+ 动作头」或「世界模型 \+ policy adapter」，本文有直接工程参考价值

        

- **建议阅读优先级**

    - 做具身智能 / VLA / 动作预测头：**精读**

    - 纯做视频/状态预测世界模型：**泛读**（重点看动作 token 化与数据混合，策略头可作为下游模块备查）

        

---



## 二、核心运作原理



### 核心思路



OpenVLA 把机器人控制问题重新表述为 **「看图说话」的续写任务**：给定一张 RGB 图像和自然语言任务指令，预训练的视觉\-语言模型（VLM）像生成文本一样，自回归地续写 N 个 action token，再经反离散化得到连续控制量。



工程类比：类似于 ChatGPT 根据 prompt 续写回答，只不过「回答」不是自然语言，而是被编码进词表末尾 256 个 token 的机器人动作序列。



### 系统主要组成模块



|模块|职责|代码位置|
|---|---|---|
|**DinoSigLIP 双视觉编码器**|DINOv2 提取空间几何特征，SigLIP 提取语义特征，channel\-wise 拼接|`prismatic/models/backbones/vision/dinosiglip_vit.py`|
|**Projector（2\-layer MLP）**|将视觉 patch embedding 投影到 Llama 2 词嵌入空间|`prismatic/models/vlms/prismatic.py`|
|**Llama 2 7B LLM Backbone**|融合视觉 token \+ 文本 prompt，自回归生成 action token|`prismatic/models/backbones/llm/llama2.py`|
|**ActionTokenizer**|连续动作 ↔ 256\-bin 离散 token 的双向转换|`prismatic/vla/action_tokenizer.py`|
|**RLDS 数据管线**|多数据集混合加载、分位数归一化、prompt 构造、loss mask|`prismatic/vla/datasets/`|
|**OpenVLA 推理封装**|`predict_action()`：图像 \+ 指令 → 反归一化连续动作|`prismatic/models/vlas/openvla.py`|
|**训练基础设施**|FSDP 分布式训练、FlashAttention、混合精度|`vla-scripts/train.py`, `prismatic/training/strategies/fsdp.py`|
|**LoRA 微调**|参数高效适配新任务/新机器人|`vla-scripts/finetune.py`|



### 数据流



```Plain Text
输入: (RGB image 224×224, language instruction)
  ↓
[Vision] 图像 → DINOv2 patches + SigLIP patches → concat → Projector → visual tokens
  ↓
[Prompt] "What action should the robot take to {instruction}?" → tokenize
  ↓
[LLM] visual tokens 插入序列 → 自回归 generate N 个 action token
  ↓
[Decode] token ID → bin center（归一化空间 [-1,1]）
  ↓
[Un-norm] 线性映射 q01/q99 → 连续 7-DoF 末端执行器增量
  ↓
输出: action vector (e.g., Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)
```



训练时数据流额外包含：RLDS 轨迹 → 分位数归一化 → ActionTokenizer 编码 → 构造 chat prompt → 仅对 action token 位置计算交叉熵损失。



### 架构示意



```Plain Text
flowchart LR
    subgraph input [Input]
        Image["RGB Image 224x224"]
        Instruction["Language Instruction"]
    end
    subgraph vision [VisionEncoder]
        DINO["DINOv2 patches"]
        SigLIP["SigLIP patches"]
        Concat["Channel Concat"]
    end
    subgraph llm [LLM Backbone]
        Projector["2-layer MLP Projector"]
        Llama["Llama 2 7B"]
    end
    subgraph output [Output]
        ActionTokens["N action tokens"]
        Continuous["Continuous 7-DoF action"]
    end
    Image --> DINO
    Image --> SigLIP
    DINO --> Concat
    SigLIP --> Concat
    Concat --> Projector
    Instruction --> Llama
    Projector --> Llama
    Llama --> ActionTokens
    ActionTokens --> Continuous
```



### 关键训练超参（论文 Section 3\.4–3\.5）



|参数|值|备注|
|---|---|---|
|参数量|7B|Prismatic\-7B VLM backbone|
|训练数据|970k 轨迹|Open X\-Embodiment 子集|
|图像分辨率|224×224|384px 无性能提升但慢 3×|
|学习率|2e\-5 固定|无 warmup|
|训练 epoch|27|远超典型 LLM 1–2 epoch|
|batch size|2048|64×A100，14 天|
|视觉编码器|**微调**（非冻结）|与 VLM 预训练惯例相反，对 VLA 性能关键|
|动作离散化|256 bins / 维度|覆盖 Llama 词表最后 256 token|



---



## 三、关键公式解析



> 论文未给出编号公式，以下根据 Section 3\.2 描述与本仓库实现还原。
> 
> 



---



### 公式：动作分位数归一化（BOUNDS\_Q99）



**原文：**



$\tilde{a}_d = \text{clip}\left(2 \cdot \frac{a_d - q_{01}^{(d)}}{q_{99}^{(d)} - q_{01}^{(d)} + \epsilon} - 1,\; -1,\; 1\right)$



**符号对照表：**



|符号|含义|类型/维度|备注|
|---|---|---|---|
|$a_d$|第 $d$ 维原始连续动作|标量|如末端执行器 x 方向增量|
|$\tilde{a}_d$|归一化后的动作|标量，$\in [-1, 1]$|训练/推理的统一输入空间|
|$q_{01}^{(d)}$|第 $d$ 维动作的 1% 分位数|标量|按数据集统计，存于 `norm_stats`|
|$q_{99}^{(d)}$|第 $d$ 维动作的 99% 分位数|标量|替代 min\-max，抑制离群值|
|$\epsilon$|数值稳定项|标量|代码中为 `1e-8`|
|$d$|动作维度索引|整数|BridgeData V2 典型为 7\-DoF|



**直觉解释：**



先把每个动作维度拉伸到 $[-1, 1]$ 区间，拉伸的上下界不是全局最小/最大值，而是 1% 和 99% 分位数。这样偶尔出现的极端动作（如传感器噪声）不会把正常动作的分布挤到很小的区间里。类似于特征工程里用分位数做 robust scaling，而不是直接 min\-max。



**公式推导：**



标准 min\-max 归一化到 $[0,1]$ 为 $(a - \min) / (\max - \min)$，再线性映射到 $[-1,1]$ 即 $2 \cdot \frac{a - \text{low}}{\text{high} - \text{low}} - 1$。将 low/high 替换为 $q_{01}$/$q_{99}$ 即得分位数版本。clip 操作防止超出分位数范围的值破坏 $[-1,1]$ 边界。



**公式应用：**



在 RLDS 数据加载阶段对每个 trajectory 的动作字段做变换（`NormalizationType.BOUNDS_Q99`）。统计量 $q_{01}, q_{99}$ 在数据集预处理时计算并持久化，推理时用同一统计量做反归一化。对应代码：



```Python
# prismatic/vla/datasets/rlds/utils/data_utils.py
tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1)
```



**容易误解的地方：**



1. **误读**：分位数归一化直接决定 256 个 bin 的边界。实际上归一化和离散化是**两步**：先归一化到 $[-1,1]$，再在固定 $[-1,1]$ 区间上做均匀 256 分箱（见公式 2）。

2. **误读**：所有维度都参与归一化。代码中 `mask` 和 `zeros_mask` 会跳过无效维度（如 min==max 的夹爪维度），这些维度直接置 0。

    

---



### 公式：均匀离散化（256 bins）



**原文：**



$b_d = \text{digitize}(\tilde{a}_d,\; \{c_0, c_1, \ldots, c_{256}\}) \in \{1, \ldots, 256\}$



其中 $\{c_i\} = \text{linspace}(-1, 1, 256)$ 为均匀 bin 边界。



**符号对照表：**



|符号|含义|类型/维度|备注|
|---|---|---|---|
|$\tilde{a}_d$|归一化动作（公式 1 输出）|标量|输入已 clip 到 $[-1,1]$|
|$b_d$|离散 bin 索引|整数|`np.digitize` 返回 $[1, 256]$|
|$c_i$|bin 边界|标量数组，长度 256|代码 `np.linspace(-1, 1, 256)`|
|bin center|反离散化用的连续值|标量|$(c_i + c_{i+1}) / 2$，共 255 个中心|



**直觉解释：**



把 $[-1, 1]$ 区间切成 256 等份，看归一化动作落在哪一格。类似于把模拟信号做均匀量化（uniform quantization），每个维度独立量化，互不影响。



**公式推导：**



RT\-2 原始方案用 min\-max 确定量化区间；OpenVLA 改进为先用分位数归一化（公式 1），再在固定 $[-1,1]$ 上均匀分箱。论文 Section 3\.2 描述为「bin width uniformly divide the interval between 1st and 99th quantile」，实现上等价于两步操作。



**公式应用：**



训练时 `ActionTokenizer.__call__()` 将连续动作编码为 token 字符串；推理时 `decode_token_ids_to_actions()` 从 bin center 查表恢复连续值。对应代码：



```Python
# prismatic/vla/action_tokenizer.py
self.bins = np.linspace(min_action, max_action, self.n_bins)  # default: 256 bins on [-1, 1]
discretized_action = np.digitize(action, self.bins)
```



**容易误解的地方：**



1. **误读**：256 个 bin 对应 256 个 bin center。实际上 $N$ 个边界产生 $N-1 = 255$ 个区间中心；代码对索引 255 做了 clip 防止越界（见 `decode_token_ids_to_actions` 注释）。

2. **误读**：离散化损失可以忽略。256\-bin 精度约 0\.008（区间宽度 2/256），对精细操作（如拧瓶盖）可能不够，论文 Section 6 也承认 dexterous 任务不如 Diffusion Policy。

    

---



### 公式：动作 token 词汇表映射



**原文：**



$t_d = V - b_d'$



其中 $V = \text{vocab\_size}$（Llama 词表大小），$b_d' = V - b_d$ 为映射后的 token ID（代码实现）。



等价描述：覆盖 Llama 词表**最后 256 个最少使用的 token**。



**符号对照表：**



|符号|含义|类型/维度|备注|
|---|---|---|---|
|$b_d$|离散 bin 索引|整数 $\in [1, 256]$|来自 `np.digitize`|
|$t_d$|LLM token ID|整数|用于 `tokenizer.decode/encode`|
|$V$|词表大小|整数|Llama 2 约 32000|
|`action_token_begin_idx`|动作 token 起始 ID|整数|$V - 257$|



**直觉解释：**



Llama tokenizer 只预留 100 个 special token 槽位，不够放 256 个动作 token。OpenVLA 选择「鸠占鹊巢」：直接复用词表末尾最少使用的 256 个 token 的位置来代表动作 bin。类似于在一个固定大小的字典里，把最不常用的 256 个词条重新定义成机器人动作。



**公式推导：**



沿用 RT\-2 \[Brohan et al\., 2023\] 的做法。BPE tokenizer 词频越低的位置在词表末尾，覆写这些位置不影响常见文本 token 的编解码。映射公式 $t = V - b$ 是 bin 索引到 token ID 的反向对应，确保 bin 0 映射到词表最后一个位置。



**公式应用：**



```Python
# prismatic/vla/action_tokenizer.py
self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))
# encode: vocab_size - discretized_action
# decode: vocab_size - action_token_ids → bin index → bin center
```



7\-DoF 动作用 7 个连续 token 表示（每个维度 1 个 token），LLM 自回归生成 `max_new_tokens=7`。



**容易误解的地方：**



1. **误读**：动作 token 是新增的 special token。实际上没有扩展词表，而是覆写现有位置；因此换用非 Llama tokenizer 需要重新验证映射假设。

2. **误读**：一个 token 编码整个动作向量。实际上每个维度独立 1 个 token，7\-DoF → 7 token 序列。

    

---



### 公式：训练目标（Action\-Only Cross\-Entropy）



**原文：**



$\mathcal{L} = -\sum_{i \in \mathcal{A}} \log p_\theta(x_i \mid x_{<i},\, I)$



其中 $\mathcal{A}$ 为 action token 在序列中的位置集合，$I$ 为输入图像，$x_i$ 为第 $i$ 个待预测的 token。



**符号对照表：**



|符号|含义|类型/维度|备注|
|---|---|---|---|
|$\theta$|模型全部可训练参数|—|含 vision encoder \+ projector \+ LLM|
|$x_{<i}$|第 $i$ 个 token 之前的上下文|token 序列|含 prompt \+ 已生成的 action token|
|$\mathcal{A}$|参与 loss 计算的位置|索引集合|仅 action token（\+ 可选 stop token）|
|`IGNORE_INDEX`|被 mask 的位置标签|$-100$|prompt 部分不参与 loss|



**直觉解释：**



标准因果语言模型的 next\-token prediction，但**只对动作部分算 loss**。prompt（图像条件 \+ 任务指令文本）的标签全部设为 \-100 让 PyTorch 忽略。类似于微调 LLM 做 JSON 输出时，只对 JSON 字段算 loss、不对用户输入算 loss。



**公式推导：**



VLM 预训练目标为 $\mathcal{L}_{\text{VLM}} = -\sum_i \log p(x_i | x_{<i}, I)$。OpenVLA 将机器人动作 $a$ 编码为 token 序列 $\{x_{|A|}\}$，作为 assistant turn 的「回答」，沿用同一目标但限制 $\mathcal{A}$ 为动作位置。这是最大似然估计（MLE）的直接应用。



**公式应用：**



数据构造（`RLDSBatchTransform`）：



```Python
# prismatic/vla/datasets/datasets.py
labels = list(input_ids)
labels[: -(len(action) + 1)] = IGNORE_INDEX  # mask prompt tokens
```



训练入口 `vla-scripts/train.py` 通过 FSDP 策略调用 `model.forward(input_ids, pixel_values, labels=labels)`，内部由 HuggingFace CausalLM 计算 masked cross\-entropy。



**容易误解的地方：**



1. **误读**：图像 token 也参与 loss。实际上 visual token 通过 `pixel_values` 注入，不在 `labels` 序列中；loss 仅覆盖文本/action token 位置。

2. **误读**：训练时预测整个 prompt \+ action。推理时只 generate action token（`max_new_tokens=action_dim`），prompt 由用户给定。

    

---



### 公式：推理反归一化



**原文：**



$a_d = 0.5 \cdot (\hat{\tilde{a}}_d + 1) \cdot (q_{99}^{(d)} - q_{01}^{(d)}) + q_{01}^{(d)}$



**符号对照表：**



|符号|含义|类型/维度|备注|
|---|---|---|---|
|$\hat{\tilde{a}}_d$|模型预测的归一化动作|标量，$\in [-1,1]$|来自 bin center 查表|
|$a_d$|最终连续动作（物理单位）|标量|送入机器人控制器|
|$q_{01}^{(d)}, q_{99}^{(d)}$|数据集动作统计|标量|由 `unnorm_key` 选择对应数据集|



**直觉解释：**



公式 1 的逆运算：把 $[-1,1]$ 的预测值线性映射回原始动作空间。$\hat{\tilde{a}} = -1$ 对应 $q_{01}$，$\hat{\tilde{a}} = +1$ 对应 $q_{99}$。类似于图像反归一化（denormalize）后送给后处理管线。



**公式推导：**



由公式 1 反解：$\tilde{a} = 2(a - q_{01})/(q_{99} - q_{01}) - 1$，移项得 $a = 0.5(\tilde{a}+1)(q_{99}-q_{01}) + q_{01}$。



**公式应用：**



```Python
# prismatic/models/vlas/openvla.py — predict_action()
normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids)
action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
```



调用时需指定 `unnorm_key`（如 `"bridge_orig"`）以选择正确的统计量。多数据集训练的模型必须通过此 key 消歧。



**容易误解的地方：**



1. **误读**：推理输出已是物理单位。模型内部先输出 $[-1,1]$ 的 bin center，必须做反归一化才能控制机器人。

2. **误读**：所有模型共用同一套统计量。不同 embodiment 的动作空间差异大，`norm_stats` 按数据集分别存储。

    

---



### 公式：LoRA 低秩适配（Section 5\.3）



**原文：**



$\Delta W = BA, \quad W' = W + \Delta W$



其中 $W \in \mathbb{R}^{d \times k}$，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。



**符号对照表：**



|符号|含义|类型/维度|备注|
|---|---|---|---|
|$W$|原始线性层权重|矩阵|LLM 中所有 linear layer|
|$\Delta W$|低秩增量|矩阵，秩 $\leq r$|微调时仅训练 $A, B$|
|$r$|LoRA rank|整数|论文推荐 $r=32$|
|$W'$|适配后权重|矩阵|推理时可合并到 $W$|



**直觉解释：**



不改动 7B 参数的主体，只在每个线性层旁路加一个小矩阵（秩 32），学一个「修正量」。类似于在预训练模型上加一个轻量 adapter，用 1\.4% 参数量达到接近全量微调的效果。



**公式推导：**



Hu et al\. \(2019\) 基于 intrinsic dimension 假设：微调所需的变化位于低维子空间。将 $\Delta W$ 分解为两个瘦矩阵之积，参数量从 $dk$ 降为 $r(d+k)$。



**公式应用：**



论文实验：LoRA $r=32$ 在 Franka\-Tabletop 上达到 68\.2% success rate，接近全量微调的 69\.7%，VRAM 从 163GB 降至 60GB。本仓库 `vla-scripts/finetune.py` 通过 HuggingFace PEFT 实现，默认 `--lora_rank 32`。



**容易误解的地方：**



1. **误读**：LoRA 只微调 LLM。论文将 LoRA 应用于所有 linear layer（含 vision encoder 路径），但 vision encoder 是否纳入取决于具体配置。

2. **误读**：rank 越大越好。论文实验 rank=32 与 rank=64 性能无显著差异。

    

---



## 四、技术优势



1. **VLM 即策略，复用 Internet\-scale 先验**

    - 优势：视觉\-语言对齐已在 LLaVA 1\.5 等数据上完成，机器人微调只需学「动作」这一新模态

    - 世界模型意义：若基座已具备视觉\-语言理解，动作头可复用同一 backbone，避免独立训练语义模块

        

2. **完全开源的全栈方案**

    - 优势：训练代码、数据混合配方、微调/量化/部署脚本全部公开，支持消融实验

    - 世界模型意义：数据混合权重（`mixtures.py`）和 RLDS 管线的工程模式可直接迁移到自研基座

        

3. **动作 token 化统一模态**

    - 优势：控制问题退化为 LM 续写，与规划/对话共享 tokenizer 和生成基础设施

    - 世界模型意义：世界模型预测的未来状态也可 token 化，与动作 token 在同一序列中联合建模

        

4. **跨 embodiment 大规模预训练**

    - 优势：970k 轨迹、29 项评测全面超越 RT\-2\-X（55B），参数仅 7B

    - 世界模型意义：多源异构数据的混合采样策略（Octo weights \+ 自定义过滤）是构建通用基座的关键工程能力

        

5. **消费级 GPU 可部署**

    - 优势：LoRA 微调单卡 A100 10–15h；4\-bit 量化推理 7GB VRAM、性能与 bf16 持平

    - 世界模型意义：降低下游实验门槛，便于快速验证「基座 \+ adapter」假设

        

---



## 五、局限性与潜在缺陷



### 论文自述（Section 6）



|局限|详情|
|---|---|
|单帧输入|不支持多相机、proprioceptive state、观测历史|
|推理速度<br>|\~6Hz（RTX 4090），不支持 50Hz 高频控制（如 ALOHA）|
|无 action chunking|每步预测单个动作，非时序动作块|
|成功率上限|多数任务 \<90%，可靠性不足|
|设计空间未充分探索|VLM 规模、co\-training（机器人\+Internet 数据）、视觉特征选择等|



### 工程推断 \[推断\]



|问题|分析|
|---|---|
|离散化精度损失|256\-bin 均匀量化对精细操作不够，论文承认 Diffusion Policy 在 dexterous 任务上轨迹更平滑|
|控制频率敏感|README 明确指出 5–10Hz 最佳；50Hz 数据需降采样，否则模型在 idle action 上「卡住」|
|数据预处理关键|Bridge 数据集 zero\-action 过滤显著影响性能（Appendix C）；不处理会导致策略冻结|
|数据混合非平凡|DROID 10% 混合权重导致 action token accuracy 持续低迷，最终从训练中移除|
|评测公平性|RT\-2\-X 在 Bridge 上需 query 第二大概率动作作为 workaround，反映闭源模型的工程摩擦|
|推理\-控制耦合|8\-bit 量化因推理慢（1\.2Hz）导致系统动力学不匹配，成功率暴跌至 58%（非量化精度问题）|



### 在世界模型基座场景下的具体挑战



- OpenVLA 不提供环境动力学模型，无法用于 model\-based planning 或 imagination rollout

- 单步策略难以与世界模型的多步预测衔接；需要额外的 action chunking 或 receding horizon 机制

- 若基座预测视频/状态，动作 token 空间与状态 token 空间的对齐方式论文未涉及，需自行设计

    

---



## 六、对团队的启发（最重要）



### 1\. 最值得复现或借鉴的技术点



**动作 token 化 \+ masked CE loss**（公式 3、4）



- 将控制头嵌入 LLM 词表，无需额外 policy head 网络

- 只需覆写词表末尾 256 token \+ 在 labels 中 mask 非动作位置

- 代码入口：`ActionTokenizer` \+ `RLDSBatchTransform`

    

**双视觉编码器融合**（DINOv2 \+ SigLIP）



- 空间特征（DINOv2）\+ 语义特征（SigLIP）channel concat

- Bridge 实验中比 SigLIP\-only 高 \~16% absolute success rate（Appendix D\.2）

- 对需要精确定位的操作（抓取、放置）尤其重要

    

**分位数归一化 \+ 均匀离散化两步策略**（公式 1、2）



- 比 RT\-2 的 min\-max 方案更 robust

- 实现简单，但效果显著；可直接用于自研动作编码器

    

### 2\. 对基座架构选型的影响



|基座路线|建议|
|---|---|
|**VLM/LLM 统一基座**|采用 patch\-as\-token \+ 动作词汇表覆盖，控制与语言共用生成框架；参考 `PrismaticVLM` 架构|
|**视频/状态世界模型**|OpenVLA 动作头作为 downstream policy adapter；世界模型负责预测 $s_{t+1}$，VLA 负责 $a_t = \pi(s_t, \ell)$|
|**混合架构**|在 world model 的 latent 空间上接轻量 action head，而非直接复用 7B LLM 做控制（降低推理成本）|



关键设计决策：

- 视觉编码器是否在下游任务中微调（OpenVLA 答案是**是**，与 VLM 预训练惯例相反）

- 动作表示：离散 token vs 连续回归 vs diffusion（OpenVLA 选离散 token，牺牲精度换统一性）

- 数据混合权重是否需要 per\-dataset 动态调整（DROID 失败案例说明不能简单叠加大数据集）

    

### 3\. 实验第一步



```Bash
# Step 1: 最小依赖安装
pip install -r requirements-min.txt

# Step 2: 零样本推理验证（无需训练数据）
python -c "
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np

processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    'openvla/openvla-7b', torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to('cuda:0')

# dummy image + instruction
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = 'In: What action should the robot take to pick up the cup?\nOut:'
inputs = processor(prompt, image).to('cuda:0', dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key='bridge_orig', do_sample=False)
print('Action shape:', action.shape)  # expect (7,)
"

# Step 3: LoRA 微调（需准备 RLDS 格式数据）
# torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
#   --vla_path openvla/openvla-7b --dataset_name <YOUR_DATASET> ...
```



**验证清单：**

1. 推理管线跑通，action shape 正确

2. 用训练集图像喂入微调后模型，复现训练日志中的 token accuracy / L1 error（README Troubleshooting 建议的 sanity check）

3. 对比 `image_aug=True/False` 对收敛速度的影响

    

### 4\. 工程暗坑（论文未明说）



|暗坑|详情|缓解|
|---|---|---|
|**Prompt 格式**|Llama 需要在 `:` 后插入 empty token（ID 29871）|推理代码 `openvla.py` L58\-64 自动处理|
|**unnorm\_key 错配**|多数据集模型必须指定统计量 key|检查 `config.json` 中 `norm_stats` 的 key|
|**center\_crop 不一致**|LIBERO 微调用 random crop 90% area，推理需 center 90% crop|`--center_crop True`|
|**推理延迟 ≠ 量化精度**|8\-bit 慢导致控制频率不匹配才是性能下降主因|优先 4\-bit 或 bf16；关注 Hz 而非仅 VRAM|
|**词表覆写假设**|最后 256 token 映射仅验证于 Llama BPE|换 tokenizer 需重新实现 `ActionTokenizer`|
|**zero\-action 污染**|训练数据首帧 zero action 导致策略冻结|过滤 no\-op transition（Appendix C）|
|**idle action 敏感**|数据中含大量「几乎不动」的步骤，推理时模型卡住|采集数据时保持连续运动（README Troubleshooting）|
|**多数据集 action space**|不同机器人 action 维度/语义不同|仅训练单臂 EEF 控制 \+ 3rd person camera 的数据|



---



## 七、相关资源



### 论文依赖的关键前置工作



- **RT\-2** \(Brohan et al\., 2023\) — VLA 范式开创者，动作 token 化方案来源

- **Prismatic VLM** \(Karamcheti et al\., 2024\) — OpenVLA 的 VLM backbone

- **Open X\-Embodiment** \(2023\) — 970k 轨迹训练数据来源

- **Octo** \(2023\) — 数据混合权重参考、主要对比基线

- **LLaVA 1\.5** \(2023\) — VLM 预训练数据配方

- **DINOv2** \(2023\) / **SigLIP** \(2023\) — 双视觉编码器

- **Llama 2** \(2023\) — LLM backbone

- **LoRA** \(Hu et al\., 2019\) — 参数高效微调

    

### 值得对比阅读的相关工作



|工作|对比维度|
|---|---|
|**RT\-2\-X** \(55B, 闭源\)|同路线 VLA，OpenVLA 参数少 8×、性能更好|
|**Octo** \(93M, 开源\)|不同架构（非端到端 VLM），OpenVLA 在语言 grounding 上更强|
|**Diffusion Policy**|数据高效模仿学习，精细操作更强但无 Internet 先验|
|**FAST** \(2025\)|动作 token 压缩，推理快 15×，对比离散化效率|
|**OpenVLA\-OFT** \(2025\)|连续动作微调，推理快 25–50×，OpenVLA 官方推荐的后续微调方案|



### 开源代码



|资源|链接|
|---|---|
|本仓库|[github\.com/openvla/openvla](https://github.com/openvla/openvla)|
|HuggingFace 模型|[openvla/openvla\-7b](https://huggingface.co/openvla/openvla-7b)|
|Prismatic 兼容 checkpoint|[openvla/openvla\-7b\-prismatic](https://huggingface.co/openvla/openvla-7b-prismatic)|
|LIBERO 微调 checkpoint|`openvla/openvla-7b-finetuned-libero-*`|
|RLDS 数据构建工具|[kpertsch/rlds\_dataset\_builder](https://github.com/kpertsch/rlds_dataset_builder)|
|项目主页|[openvla\.github\.io](https://openvla.github.io)|



---



## 附录：论文描述与代码实现交叉验证



> 以下标注论文与代码的已知差异，供实现时参考。
> 
> 



|主题|论文描述|代码实现|一致性|
|---|---|---|---|
|动作离散化区间|bin width 按 1%/99% quantile 均匀划分|先 BOUNDS\_Q99 归一化到 $[-1,1]$，再 `linspace(-1,1,256)` 均匀分箱|**等价但两步实现**|
|分位数|1st/99th quantile|代码用 `q01`/`q99`（1%/99%）|**一致**|
|Token 映射|覆写最后 256 个最少使用 token|`vocab_size - discretized_action`|**一致**|
|Loss mask|仅 action token 计算 CE|`labels[:-(len(action)+1)] = IGNORE_INDEX`|**一致**|
|视觉编码器|微调（非冻结）|FSDP 训练默认含 vision backbone 参数|**一致**|
|推理 prompt|chat 格式指令|`"What action should the robot take to {instruction}?"`|**一致**|
|反归一化|未给显式公式|`0.5 * (norm + 1) * (q99 - q01) + q01`|**一致**（公式 1 逆运算）|
|训练 epoch|27 epoch|配置于 `prismatic/conf/vla.py`|**一致**|
|图像分辨率|224px|`dinosiglip-vit-so-224px`|**一致**|
|DROID 数据|尝试过 10% 混合后移除|`mixtures.py` 中最终配方不含 DROID|**一致**|
|ActionTokenizer 默认范围|按数据集 quantile|`ActionTokenizer` 默认 `min_action=-1, max_action=1`|**一致**（输入已预归一化）|



**信息不足、建议进一步验证：**

- 论文 27 epoch 对应的 exact checkpoint step 与 `openvla-7b` HF 权重是否完全对齐（需对比 `config.json` 中 `step` 字段）

- OFT/FAST 后续方案与本文 ActionTokenizer 的兼容性（超出本文范围）



Copyright © 2026 [Austin-152](https://github.com/Austin-152)\. All rights reserved\.



