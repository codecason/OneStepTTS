# pMF 接入 WavTTS：概念与待办

## u、v、vc、vg 是什么

这四个量都是速度，可以理解为“带噪数据应该怎样变化”。它们的区别是来源和用途。

| 符号 | 含义 | 用途 |
|---|---|---|
| `u` | u 头预测的平均速度：从时间 t 到 r，这段区间应该怎么走 | 生成时使用，支持一步跨越较大的时间区间 |
| `v` | v 头预测的瞬时速度：当前这一刻应该怎么走 | 训练时接受瞬时速度目标的监督 |
| `vc` | 有条件输入下，调用 v 头得到的瞬时速度预测；c 表示 conditional | 构造引导目标，并为 JVP 提供运动方向 |
| `vg` | 根据真实样本和模型预测构造的引导速度目标；g 表示 guided | 作为训练监督目标，不是独立的网络输出头 |

**网络只有 u、v 两个输出头。vc 是 v 头在构造目标时的一次预测，vg 是计算出来的目标。** 最终损失里的 v 与 vc 来自不同的网络调用，输入条件也可能不同，不能直接认为它们数值相等。

## 一次训练中如何使用

```text
先调用 v 头
    ↓
得到有条件预测 vc，以及无条件预测 vu
    ↓
结合真实样本构造引导目标 vg
    ↓
用 vc 作为 JVP 的运动方向，再运行网络得到 u、du_dt、v
    ↓
用 vg 监督两项：
    V = u + (t-r) × stop_gradient(du_dt)  → 接近 vg
    v                                    → 接近 vg
```

忽略引导区间和条件丢弃的分支时，目标构造为：

$$
v_g=v_t+\left(1-\frac{1}{\omega}\right)(v_c-v_u).
$$

其中，`vt` 是由真实样本和带噪样本计算的基础速度目标，`vu` 是无条件预测，`omega` 是引导强度。实际代码会根据引导区间调整有效强度；条件丢弃时，对应的 vg 改为 vt。

主分支并不直接要求 u 接近 vg，而是要求：

$$
V=u+(t-r)\operatorname{stopgrad}\left(\frac{du}{dt}\right)
\approx \operatorname{stopgrad}(v_g).
$$

原因是 u 描述整段时间的平均速度，vg 描述当前时刻的瞬时速度目标，中间需要导数修正。这里的 du/dt 是沿 `(vc, 1, 0)` 方向对输入 `(z,t,r)` 做 JVP 得到的变化率，不只是对时间变量求偏导。

vg 和 JVP 方向 vc 不参与反向传播；最终预测 u、v 保留参数梯度。

## 生成时使用什么

按当前 pMF 的时间约定，t=1 为噪声端，t=0 为数据端。每一步只需要平均速度 u：

$$
z_r=z_t-(t-r)u.
$$

生成时不计算 JVP，也不需要构造 vg。v 头主要用于训练。

## 接入 WavTTS 时的形状与待办

当前 WavTTS 的波形接口是 `[B,N]`：B 为 batch 大小，N 为补齐后的采样点数量。因此接入后，u、v、vc、vg 都可以保持 `[B,N]`，时间用于波形广播时为 `[B,1]`。

这是接入设计说明；当前 WavTTS 尚未改成 pMF。

- [ ] 以现有 `dit.py` 为基础，在新 `src/wavtts/model/backbones/pmf_dit.py` 中增加 u/v 双分支及时间跨度、引导参数条件，保留文本与参考音频条件。
- [ ] 在新 `src/wavtts/model/pmf.py` 的训练前向中构造 vc、vg、JVP 和双分支损失，并保留有效长度与生成区域 mask。
- [ ] 在同一文件的采样函数中改为平均速度更新，支持一步／少步生成。
- [ ] 统一时间方向：当前 WavTTS 是噪声端 0 → 数据端 1，当前 pMF 是噪声端 1 → 数据端 0，不能直接混用速度符号和采样公式。
- [ ] 将条件引导从图像类别适配为文本与参考音频；明确有条件、无条件分支分别保留或丢弃哪些输入。

## 函数框架草案

状态：**已确定接口和调用关系，未实现完整 pMF 训练。** 新代码放在独立模块，现有 CFM 训练和推理入口保持原样。未实现的辅助方法抛出 `NotImplementedError`。
- [model/pmf.py](src/wavtts/model/pmf.py)：`PMF`，负责训练目标、速度换算和采样循环。
- [backbones/pmf_dit.py](src/wavtts/model/backbones/pmf_dit.py)：`PMFDiT`，负责条件输入、共享网络和双分支输出。

### 先确定的约定

| 项目 | 本阶段约定 |
|---|---|
| 时间方向 | 采用 pMF：噪声 t=1，干净波形 t=0；不复用旧 CFM 的 `_x_to_v` |
| 外部波形 | `[B,N]`，未缩放；网络内部仍使用 WavTTS 的幅度缩放 |
| 时间与引导条件 | `[B]`，乘波形时使用 `[:,None]` 广播 |
| 网络直接输出 | `PMFDiT` 返回两幅缩放后的干净波形预测 `x_u,x_v`；只有 `predict_velocity` 将它们换算成 u、v |
| 网络结构 | 保留 WavTTS 文本、参考音频、一维位置编码；保留 t 调制并增加 h、omega、区间调制；不照搬图像 pMF 的前缀 token |
| 条件丢弃 | 按样本联合丢弃文本和参考音频；一次采样后固定，JVP 内不再抽样或修改缓存 |
| padding 与参考区域 | padding 不参与损失，padding 处速度/JVP 方向为零；保留参考区域作为条件，最终输出恢复参考波形，不在每步强行覆盖状态 |
| 主损失归约 | 初版按每条音频的预测区域求误差和，再做 pMF 自适应加权，最后按有效样本平均；空预测区域排除。后续需验证音频长度对权重的影响 |
| 感知损失 | 初版保留 WavTTS 原有 masked Mel loss 及其权重，不增加图像感知网络，也不立即对 Mel 引入自适应加权 |
| 初始化与优化器 | 双头初始化及旧权重映射另行实现；先沿用 WavTTS 的 AdamW/EMA，不同时迁移 Muon |

`PMFOptions` 中的数值只是起始设计，不是经过实验验证的配置。`P_mean=+0.8` 是将原 WavTTS 时间改为 `1-t` 后，对原 `P_mean=-0.8` 的符号转换；增加时间对和引导训练后仍需调参。

### 一、核心函数伪代码

下面是函数式框架，`model` 是 PMF 对象，辅助方法对应代码中的同名方法。已有 Trainer 负责反向传播和状态更新；实际代码暂未将它的模型构造切换为 PMF。

```python
# 顶端训练入口：train_step
# 输入：PMF 模型、训练状态、(wave, text, lengths)
# 输出：更新后的训练状态、日志指标

def train_step(model, state, batch):
    loss, _, _, metrics = forward(model, *batch)
    grads = backward(loss, model.parameters())
    state = update_optimizer_and_ema(state, grads)
    return state, metrics

def forward(model, wave, text, lengths):
    # 输入：未缩放波形[B,N]、文本、真实长度[B]
    # 输出：标量loss、条件波形[B,N]、平均速度u[B,N]、日志字典
    x, cond = model.prepare_training_batch(wave, text, lengths)
    s = model.sample_schedule(x.shape[0], x.device)
    z, vt = model.make_noisy_pair(x, cond.valid_mask, s.t)
    vg, vc = model.build_guided_target(z, vt, cond, s)
    cond, vg = model.apply_condition_dropout(cond, vg, vt)
    u, du_dt, v = predict_with_jvp(model, z, cond, s, vc)
    V = u + (s.t-s.r)[:,None] * stop_gradient(du_dt)
    Lu, Lv = model.compute_flow_losses(V, v, stop_gradient(vg), cond)
    mel = model.compute_mel_loss(z-s.t[:,None]*u, x, cond, s.t)
    return Lu+Lv+mel, cond.audio, u, {
        "total_loss": Lu+Lv+mel, "flow_loss": Lu+Lv,
        "loss_u": Lu, "loss_v": Lv, "aux_mel_loss": mel,
    }

# 顶端推理入口：sample
# 输入：参考音频、文本、目标采样点数，以及步数/引导参数
# 输出：未缩放的生成波形[B,Nmax]、缩放空间的采样轨迹

def sample(model, reference, text, duration, lengths, seed,
           steps, omega, tmin, tmax):
    z, cond = model.prepare_sampling(reference, text, duration, lengths, seed)
    times = linspace(1, 0, steps+1)
    trajectory = [z]
    for t, r in zip(times[:-1], times[1:]):
        z = sample_step(model, z, cond, t, r, omega, tmin, tmax)
        trajectory.append(z)
    return model.finalize_waveform(z, cond), stack(trajectory)

def sample_step(model, z, cond, t, r, omega, tmin, tmax):
    # 输入：当前波形[B,N]、条件、标量时间对和引导参数
    # 输出：下一时刻波形[B,N]
    t, r, omega, tmin, tmax = broadcast_to_batch(z, t, r, omega, tmin, tmax)
    u, _ = predict_velocity(model, z, cond, t, t-r, omega, tmin, tmax)
    return z - (t-r)[:,None]*u
```

`backward` 输入 loss 与参数、输出梯度；`update_optimizer_and_ema` 输入旧状态与梯度、输出新状态，包含梯度归约、裁剪、优化器/学习率/EMA 更新；两者由现有 Trainer 承担。`broadcast_to_batch` 将标量转换成与 z 同设备的 `[B]` 张量。其它张量原语不展开实现。

### 二、辅助函数与接口

先写出直接关系到数学含义的两个辅助函数。其余辅助步骤仅规定输入输出，真实代码中已建立对应的待实现方法。

```python
# 被 forward 调用

def predict_with_jvp(model, z, cond, schedule, vc):
    # 输入：状态[B,N]、固定条件、时间表、方向vc[B,N]
    # 输出：u、du_dt、v，均为[B,N]
    field = lambda z,t,r: predict_velocity(
        model, z, cond, t, t-r,
        schedule.omega, schedule.t_min, schedule.t_max)
    return jvp_with_aux(
        field, (z,schedule.t,schedule.r),
        (stop_gradient(vc), ones_like(schedule.t), zeros_like(schedule.r)))

# 被 predict_with_jvp 和 sample_step 调用；也供引导目标构造使用

def predict_velocity(model, z, cond, t, h, omega, tmin, tmax):
    # 输入：带噪波形[B,N]、固定条件、标量条件[B]
    # 输出：u、v两个速度[B,N]
    xu, xv = model.transformer(
        x=z, cond=cond.audio, text=cond.text, time=t, h=h,
        omega=omega, t_min=tmin, t_max=tmax,
        mask=cond.valid_mask, lens=cond.lengths, drop_mask=cond.drop_mask)
    denom = clip(t, model.options.t_eps, 1)[:,None]
    u, v = (z-xu)/denom, (z-xv)/denom
    return zero_padding(u, cond.valid_mask), zero_padding(v, cond.valid_mask)
```

`jvp_with_aux` 的输入为函数、输入点和方向，输出 `(u, J_u·方向, v)`；`zero_padding` 输入张量与有效位置 mask，将无效位置置零并返回同形状张量。

| 方法 | 输入 → 输出 | 主要职责 |
|---|---|---|
| `prepare_training_batch` | 原波形、文本、长度 → 缩放干净波形 x、`WaveCondition` | 文本编码、随机预测区域、参考条件与 mask |
| `sample_schedule` | batch 大小、设备 → `FlowSchedule` | t/r、FM mask、omega 和引导区间；所有字段为 `[B]` |
| `make_noisy_pair` | x、有效 mask、t → z、vt，均为 `[B,N]` | 加噪及截断分母的基础速度目标 |
| `build_guided_target` | z、vt、条件、时间表 → 停止梯度的 vg、vc | 调用 v 分支构造条件/无条件引导，处理 FM 和引导区间 |
| `apply_condition_dropout` | 条件、vg、vt → 固定丢弃 mask 的新条件、新 vg | 同时丢弃音频与文本；丢弃样本目标改为 vt，保留之前算出的 vc |
| `compute_flow_losses` | V、v、vg、条件 mask → 两个标量损失 | 按有效预测区域计算双分支目标及自适应权重 |
| `compute_mel_loss` | 预测/真实缩放波形、条件、t → 标量 | 还原幅度后计算原 WavTTS Mel loss；t 预留供后续区间策略使用 |
| `prepare_sampling` | 参考音频、文本、时长、参考长度、seed → 初始噪声、新条件 | 参考音频 padding、文本处理、采样长度与噪声 |
| `finalize_waveform` | 最终缩放波形、条件 → 未缩放波形 | 恢复参考区域、还原幅度、清零长度之外的位置 |
| `PMFDiT.forward` | 波形/参考/文本、t/h/引导条件、mask → xu、xv `[B,N]` | 双分支预测干净波形，不在此处转换速度 |
| `prepare_tokens` | 波形/参考/文本、mask/lengths/drop_mask → token、token mask、RoPE | 复用 WavTTS 一维波形分帧与文本前端 |
| `embed_flow_conditions` | t/h/omega/tmin/tmax `[B]` → 调制向量 `[B,D]` | 融合时间、跨度与引导信息 |
| `run_shared_blocks` | token、调制、mask/RoPE → 共享特征 `[B,S,D]` | 共享 Transformer 主干 |
| `run_output_heads` | 共享特征、调制、mask/RoPE → 两路波形块 `[B,S,frame_len]` | 独立 u/v 分支及输出投影 |
| `unpatchify_heads` | 两路波形块、原始采样点数 → xu、xv `[B,N]` | 还原并裁剪到原始波形长度 |

主调用链：

```text
训练：Trainer → PMF.forward
  → 准备数据/采样时间/构造引导目标
  → predict_with_jvp → predict_velocity → PMFDiT.forward
  → 双分支速度损失 + Mel损失 → Trainer反向与更新

生成：PMF.sample → sample_step → predict_velocity → PMFDiT.forward
  → finalize_waveform

网络：PMFDiT.forward → prepare_tokens + embed_flow_conditions
  → run_shared_blocks → run_output_heads → unpatchify_heads
```

### 后续接线任务

- [x] 分离核心/辅助函数，建立输入输出与波形单位契约。
- [x] 添加独立 `PMF`、`PMFDiT` 骨架，未完成路径明确报错。
- [ ] 实现辅助方法和双分支模块；先验证小波形的 JVP、mask 和速度符号。
- [ ] 验证注意力后端支持 JVP；先关闭 activation checkpoint 和训练随机 dropout，禁用 JVP 内缓存写入。
- [ ] 在 `model/__init__.py` 导出新类，并在 `train/train.py` 增加模型选择及配置；目前不启用。
- [ ] 接入 Trainer 的分支损失日志、混合精度/分布式验证、权重初始化与 checkpoint 兼容策略。
- [ ] 同步推理和批量评估入口，显式传入 `omega/t_min/t_max`；不得静默把旧 `cfg_strength` 当成 omega。
- [ ] 对比原 WavTTS 的训练与音质指标，再决定是否调整 Mel、自适应损失归约或优化器。
