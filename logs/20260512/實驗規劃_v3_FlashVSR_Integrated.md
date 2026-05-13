# FlashVSR_Integrated PTQ 量化實驗規劃（v3）
## 目標：A8W8（研究哪些模塊可用 A16）讓 PSNR 越接近 30越好

**Author:** localpc  
**Date:** 2026-05-06  
**Project:** `/home/user/apps/FlashVSRptq/FlashVSR_Integrated/`  
**Output:** 寫入 `~/Dropbox/teamResearch/FlashVSR-PTQ/`  
**Git Log 最新結果：** W8A8 SmoothQuant → **9.07 dB PSNR** vs baseline（需要大幅改善）

---

## 1. 現有基礎分析

### 1.1 當前量化現況

| 模塊 | 目前狀態 | 問題 |
|------|---------|------|
| `WeightOnlyInt8Linear` | W8A16，per-channel int8 weight + fp16 activation | 只有 weight 量化，activation 仍是 fp16 |
| `SmoothQuantLinear` | W8A8，migration factor + per-channel act scale | 9.07 dB 說明 activation 量化誤差大 |
| `convert_model_to_w8a8_smoothquant` | 替換所有 `nn.Linear` → `SmoothQuantLinear` | 沒有針對 attention/FFN 差異化處理 |
| `ObserverLinear` + `inject_observers` | 收集 `act_amax` 用於 calibration | amax 收集方式不夠精確（max 而非 MSE/percentile） |

### 1.2 WanModel 架構（DiT，30 層）
```
WanModel(dim=1536, num_heads=12, num_layers=30, ffn_dim=6144)
├── 30 × DiTBlock
│   ├── SelfAttention (q,k,v,o = nn.Linear, norm_q, norm_k)
│   │   └── 4× nn.Linear: q, k, v, o  (各 dim→dim = 1536→1536)
│   ├── CrossAttention (q,k,v,o = nn.Linear, norm_q, norm_k)
│   │   └── 4× nn.Linear: q, k, v, o
│   ├── FFN: nn.Sequential(Linear→GELU→Linear)
│   │   └── 2× nn.Linear: dim→ffn_dim, ffn_dim→dim
│   └── modulation: nn.Parameter(1, 6, dim)
└── Head (norm + Linear → output)
    └── 1× nn.Linear: dim→out_dim*patch
```
**核心量化目標：** SelfAttention.Q/K/V/O + CrossAttention.Q/K/V/O + FFN.0/1

### 1.3 為什麼目前只有 9.07 dB

1. **Activation quantization 誤差傳播**：Diffusion model forward pass 有 timestep-dependent activation distribution，static per-channel scale 無法適配
2. **SmoothQuant alpha=0.5 固定**：alpha 影響 migration factor，但沒有根據各層 activation 統計調整
3. **沒有區分 attention 層 vs FFN 層**：attention 的 SoftMax(QK)V 流程對 activation outlier 非常敏感，FFN 的 GELU 變化相對平穩
4. **ObserverLinear 用 max 而非 percentile**：max 容易被單一 outlier 支配，MSE-optimal scale 更好

---

## 2. SOTA 參考文獻

### 必讀（直接相關）

| Paper | Venue | 核心貢獻 | 適用於 FlashVSR |
|-------|-------|----------|----------------|
| **QuantVSR** (arXiv:2508.04485) | 2026 | 低位元 VSR 量化：STCA（spatio-temporal complexity aware）+ LBA（learnable bias alignment） | 最高！直接針對 VSR，基於 MGLD-VSR（FlashVSR 同源架構） |
| **PMQ-VE** (arXiv:2505.12266, NeurIPS 2025) | NeurIPS 2025 | Progressive Multi-Frame Quantization：coarse-to-fine temporal modeling | Frame-wise dynamic range adaptation |
| **PTQ4DiT** (NeurIPS 2024) | NeurIPS 2024 | CSB (Channel Salience Balancing) + SSC (Spearman Salience Calibration) | CSB 概念可直接用於 WanModel attention layers |
| **ViDiT-Q** (ICLR 2025) | ICLR 2025 | W8A8 INT8 for DiT：rotation-based activation outlier 处理 | 證明 W8A8 在 DiT 上可做到 negligible drop |
| **DVD-Quant** (arXiv:2505.18663) | 2025 | δ-GBS (dynamic Bounded Grid Search) + ARQ (Auto-scaling Rotated Quantization) | Video DiT 動態 activation bound 估計 |

### 輔助參考

| Paper | Venue | 核心貢獻 |
|-------|-------|----------|
| **TFMQ-DM** (CVPR 2024 Highlight) | CVPR 2024 | Temporal Feature Maintenance：timestep grouping calibration |
| **Q-DiT4SR** (arXiv:2602.01273) | 2025 | VaSMP (Variance-aware Spatial Mixed Precision) + H-SVD |
| **AWQ** (NeurIPS 2024) | NeurIPS 2024 | Activation-aware weight quantization：根據 act_amax 保護顯著權重 |
| **QuantSR** (NeurIPS 2023) | NeurIPS 2023 | Low-bit quantization for image super-resolution |

---

## 3. PR 規劃

### PR #0（緊急）：診斷與基準建立
**目標：** 搞清楚 9.07 dB 的瓶頸到底在哪一層

**內容：**
- [ ] 建立 per-layer PSNR 分析：分別測量每一層被量化後對輸出的影響（ablation）
- [ ] 確認哪些層的量化誤差最大（預期：SelfAttention.Q/K/V + CrossAttention.Q/K/V）
- [ ] 分析 activation distribution：各層 amax 的 coefficient of variation（CV），CV > 1 的層不適合 static quantization
- [ ] 建立公平的基準：先用 W8A16 (`WeightOnlyInt8Linear`) 作為上界參照

**預期產出：** 瓶頸層排名、CV 統計表

---

### PR #1：Temporal-Aware Activation Calibration（基於 QuantVSR + PMQ-VE）
**目標：** 解決 frame-wise temporal variation 造成的 activation scale 不準問題

**參考 QuantVSR 的 STCA 機制：**
- 測量 spatial complexity（單幀內 activation range）和 temporal complexity（跨幀 activation range 變化）
- 根據 complexity 分配不同的 rank/capacity 給 auxiliary FP branch

**FlashVSR 適配：**
- [ ] 改進 `collect_activation_stats`：不只是 max，還要記錄 `mean, std, percentile(99/95/90)`
- [ ] 計算 temporal complexity score：`CV = std(amax_per_frame) / mean(amax_per_frame)`
- [ ] 對高 CV 層（> 0.5）：標記為需要 dynamic scale 或 FP32 residual
- [ ] **Per-frame activation profile**：記錄每個 calibration video 的每一幀激活範圍，識別最壞情况

**驗收：** 有/無 temporal-aware calibration 的 PSNR 對比（預期 +1~3 dB）

---

### PR #2：CSB (Channel Salience Balancing) for WanModel Attention
**目標：** 降低 SelfAttention/CrossAttention 層的量化誤差（PTQ4DiT）

**背景：**
- PTQ4DiT 發現：activation salient channel 和 weight salient channel 通常不重疊（complementarity property）
- 透過 Salience Balancing Matrix 將雙方的 extreme 值重新分配
- 在 WanModel SelfAttention 上：Q/K/V/O 各自有 1536×1536 weight，值得實施

**內容：**
- [ ] **Salience 計算**：對 SelfAttention.Q/K/V/O 的 weight 和 input activation 計算 channel-wise salience score
- [ ] **Balancing Matrix**：`S = act_amax^alpha / weight_amax^(1-alpha)`（已有但 alpha 固定）
  - 改用 PTQ4DiT 的 complementarity-aware alpha：根據 act/weight salience overlap 動態調整 alpha
- [ ] **Per-head QKV split**：WanModel dim=1536, num_heads=12 → 每個 head 128 ch
  - QKV split calibration：Q、K、V 各自獨立收集 activation stats，避免 cross-head range 干擾
- [ ] **CSB re-parameterization**：將 migration factor 預先融合進 weight，避免 runtime 額外計算

**驗收：**
- 在 SelfAttention 層確認 CSB 前後 quantization error 下降
- Per-head QKV split 後確認 PSNR 改善

---

### PR #3：FFN INT8 + 殘差路徑處理
**目標：** FFN 層做 INT8 幾乎無 drop

**內容：**
- [ ] **FFN 層特殊性質**：GELU 的 activation distribution 比較穩定，INT8 友好
- [ ] **Per-layer vs Per-tensor**：比較 FFN 的 weight per-channel vs per-tensor 量化 PSNR
- [ ] **殘差加法對齊**：DiTBlock 中的 `x = x + self_attn_output` 是加法
  - 問題：x 是 FP16，self_attn_output 是 INT8 反量化，scale 必須對齊
  - 方案 A：fold scale into residual（re-parameterize 到 weight scale）
  - 方案 B：維持 residual path FP16，只量化主路徑
- [ ] **Modulation parameter**：`t_mod = self.modulation + t_emb`，這些參數應該保持 FP32

**驗收：** FFN 量化後確認 PSNR drop < 0.1 dB

---

### PR #4：A16 策略研究（哪些模塊適合 A16）
**目標：** 識別哪些模組不適合 A8 需要 A16

**分析框架：**
- [ ] **A16 候選模組識別原則：**
  1. Activation distribution 高 CV（temporal variation 大）→ A16
  2. 涉及 softmax / exp 計算（numerical sensitive）→ A16
  3. 跨 frame 累計的 activation（如 KV cache 相關）→ A16
- [ ] **候選模組：**
  - SelfAttention QKV：softmax(QK^T)V 對 activation outlier 敏感 → 建議 A16
  - CrossAttention QKV：同上 → 建議 A16
  - FFN intermediate（GELU 後）：GELU 是連續函數，INT8 足夠 → 建議 A8
  - LayerNorm / RMSNorm：input activation 做歸一化，distribution 相對穩定 → A8 可行
  - Head output：輸出層對精度敏感 → 建議 A16
- [ ] **實驗：** 交叉對比 W8A8 / W8A16 / W16A8 / W16A16 在不同層的組合

**驗收：** 找到最優 A8W8/A16 配置，PSNR 接近 30 dB

---

### PR #4b：非 SmoothQuant W8A8 方案
**目標：** 擺脫 SmoothQuant，改用其他 W8A8 方法

**背景：**
- SmoothQuant 的 activation migration 在 FlashVSR 上失敗（9.71 dB vs 38.37 dB W8A16）
- 即使 1 層 W8A8 也導致 23 dB 暴跌
- 需要嘗試其他 W8A8 方案

**方向 1：Per-Head QKV Split（無 SmoothQuant）**
- WanModel dim=1536, num_heads=12 → 每個 head 128 ch
- Q、K、V 各自獨立收集 activation stats，避免 cross-head range 干擾
- 每個 head 有獨立的 act_scale，不做 migration

**方向 2：ViDiT-Q Rotation-Based Outlier Handling**
- 核心思想：對 activation 執行旋轉，將 outliers 均勻分散到各 channel
- 旋轉矩陣 P 滿足 P^T P = I，選擇使得所有 channel 的 activation range 相近的 P

**方向 3：Dynamic Per-Token Activation Scales**
- 靜態 per-layer scale → 動態 per-token scale
- 每個 token 有獨立的 activation scale，更精確但需要特殊的 INT8 kernel

**驗收：** 非 SmoothQuant W8A8 PSNR > 20 dB（目標 > 30 dB）

---

### PR #5：MXFP8（Microscaling FP8）量化
**目標：** 探索 MXFP8 格式，看是否能用於 FlashVSR 的 W8A8 量化

**背景：**
- MXFP8 是 NVIDIA Hopper+ GPU 的硬體支援格式
- 使用 **E8M0 block floating point**：每 32 個連續元素共用一個 exponent scale factor
- 比 per-tensor 或 per-channel 的 static quantization 更精細
- 適合 activation（per-token scale）也適合 weights（per-block scale）

**MXFP8 格式：**
| 格式 | 總位元 | 構成 |
|------|--------|------|
| MXFP8 (E8M0) | 8 bits | 1 shared exponent (E8M0) + 32× 7-bit mantissa elements |
| MXFP6 (E3M0) | 6 bits | 1 shared exponent (E3M0) + 32× mantissa |
| MXFP4 (E2M0) | 4 bits | 1 shared exponent (E2M0) + 32× mantissa |

**Block Floating Point 原理：**
```
每 32 個元素共用一個 scale = 2^exponent
實際值 = sign × (0.mantissa) × 2^exponent
```

**FlashVSR 適配：**
- [ ] 實現 MXFP8 格式：每 32 元素 block 一個 scale factor
- [ ] 對 attention QKV/O 和 FFN 應用 MXFP8，看是否有改善
- [ ] 比較 MXFP8 vs SmoothQuant vs per-channel INT8 的 PSNR

**NVIDIA Transformer Engine 支援：**
- Hopper+ 原生支援 MXFP8，無需軟體實現
- 可以透過 `transformer_engine.pytorch.mxfp8` 包使用

**驗收：** MXFP8 W8A8 PSNR > 30 dB vs FP16

---

### PR #6：端到端 PTQ 整合驗證與 Benchmark
**目標：** 完整測量，最終目標 PSNR > 30 dB vs FP16 baseline

**內容：**
- [ ] 整合所有 PR 的最佳配置（CSB + per-head QKV + FFN A8 + A16 策略）
- [ ] 在測試影片上測量：PSNR / SSIM / LPIPS
- [ ] 測量 inference latency（ms/frame）
- [ ] 測量 VRAM usage vs FP16 baseline
- [ ] Ablation study：各技術單獨貢獻（CSB only / temporal-aware only / mixed-precision only）
- [ ] 寫入 benchmark 結果到 `~/Dropbox/Daily/flashvsr_ptq_results.md`

**驗收標準：**
- PSNR vs FP16 baseline drop < 0.2 dB（目標 < 0.1 dB）
- VRAM 節省 > 30%（相對於 FP16）

---

## 4. 技術路線圖

```
現狀：W8A8 SmoothQuant → 9.07 dB（需要大幅改善）

Phase 1（PR #0 + PR #1）：診斷 + Temporal-aware calibration
  PR#0 → per-layer ablation 確認瓶頸在哪
  PR#1 → activation calibration 改用 percentile + temporal profile

Phase 2（PR #2 + PR #3）：SmoothQuant 失敗，轉向非 SmoothQuant W8A8
  PR#2 → Mixed-precision A16（確認 SmoothQuant 不可行）
  PR#2b → 非 SmoothQuant W8A8：per-head QKV / ViDiT-Q rotation / dynamic scales

Phase 3（PR #4 + PR #4b）：A16 策略 + 其他 W8A8 方案
  PR#4 → 找出哪些模塊要用 A16
  PR#4b → Per-head QKV split / ViDiT-Q rotation / dynamic per-token scales

Phase 4（PR #5 + PR #6）：MXFP8 + 端到端驗證
  PR#5 → MXFP8 block floating point 格式探索
  PR#6 → 完整 benchmark + ablation
```

---

## 5. 關鍵風險與對策（已更新 2026-05-06）

| 風險 | 說明 | 對策 |
|------|------|------|
| SmoothQuant activation migration 失敗 | PR #2 證實：即使 1 層 W8A8 也導致 23 dB 暴跌（38.37→9.71 dB） | 放棄 SmoothQuant，嘗試 non-SmoothQuant W8A8（per-head QKV split、ViDiT-Q rotation、dynamic scales） |
| Temporal-aware calibration 不夠 | PR #1 證實：temporal CV 很小（0.03-0.06），改善 < 0.2 dB | 問題是結構性的，不是 temporal variation |
| Diffusion forward 誤差傳播 | 每層量化誤差在 30 層中累積放大 | Layer-wise ablation 確認最大誤差來源，W8A16 作為安全 baseline |
| A8 softmax 精度不足 | softmax(QK^T)V 中 Q/K/V 用 INT8 時精度不足 | 參考 ViDiT-Q 的 rotation-based outlier handling 或保持 QKV A16 |
| 9.07 dB 根本問題 | activation 量化在 FlashVSR 結構性不適用 | 需要非 SmoothQuant 方案，或全面使用 W8A16 |
| MXFP8 硬體依賴 | MXFP8 需要 NVIDIA Hopper+ GPU，軟體實現複雜 | 先嘗試軟體實現的 block floating point，或等待硬體支援 |

---

## 6. 參考文獻（完整）

1. **QuantVSR** — "Low-Bit Post-Training Quantization for Real-World Video Super-Resolution" (arXiv:2508.04485, 2026)
   - 基於 MGLD-VSR（FlashVSR 同源），直接針對 VSR 量化
   - STCA：spatio-temporal complexity aware mechanism
   - LBA：learnable bias alignment module

2. **PMQ-VE** — "Progressive Multi-Frame Quantization for Video Enhancement" (arXiv:2505.12266, NeurIPS 2025)
   - Progressive coarse-to-fine two-stage quantization
   - Frame-wise dynamic range adaptation

3. **PTQ4DiT** — "PTQ4DiT: Post-Training Quantization for Diffusion Transformers" (NeurIPS 2024)
   - CSB: Channel Salience Balancing
   - SSC: Spearman-guided Salience Calibration
   - Code: https://github.com/adreamwu/PTQ4DiT

4. **ViDiT-Q** — "ViDiT-Q: Efficient Post-Training Quantization for DiT-based Generative Models" (ICLR 2025)
   - W8A8 for DiT with negligible drop
   - Rotation-based activation outlier handling

5. **DVD-Quant** — "Depthwise Video Diffusion Quantization" (arXiv:2505.18663, 2025)
   - δ-GBS: dynamic Bounded Grid Search
   - ARQ: Auto-scaling Rotated Quantization
   - δ-GBS: temporal-aware bounds

6. **TFMQ-DM** — "Temporal Feature Maintenance for Diffusion Models" (CVPR 2024 Highlight)
   - Timestep grouping calibration
   - Code: https://github.com/ModelTC/TFMQ-DM

7. **AWQ** — "AWQ: Activation-aware Weight Quantization for LLM Compression" (NeurIPS 2024)
   - 根據 activation magnitude 保護重要權重

8. **QuantSR** — "QuantSR: Accurate Low-bit Quantization for Efficient Image Super-Resolution" (NeurIPS 2023)
   - 低位元量化 for 超解析度網路
