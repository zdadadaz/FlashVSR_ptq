# FlashVSR_Integrated PTQ 量化實驗規劃（v3）日誌

## 日期：2026-05-06 09:58 GMT+8

---

## 已完成

### 研究
- [x] 詳細閱讀 `/home/user/apps/FlashVSRptq/FlashVSR_Integrated/` 專案架構
- [x] 分析 `src/models/quantization/quant.py`：WeightOnlyInt8Linear、SmoothQuantLinear
- [x] 分析 `src/models/quantization/smoothquant.py`：ObserverLinear、inject_observers、collect_activation_stats
- [x] 分析 `scripts/ptq_test.py`、`ptq_calibrate.py`、`ptq_convert_w8a16.py`
- [x] 分析 `nodes.py` 中 `init_pipeline` + quantize_mode 整合邏輯
- [x] 分析 `wan_video_dit.py` WanModel + DiTBlock 架構（30層 SelfAttention+CrossAttention+FFN）
- [x] 網路搜尋 SOTA 量化論文：QuantVSR、PMQ-VE、PTQ4DiT、ViDiT-Q、DVD-Quant
- [x] 讀取團隊既有的 `FlashVSR-PTQ/實驗規劃.md`（v2版本）

### 產出
- [x] 寫入 `~/Dropbox/teamResearch/FlashVSR-PTQ/實驗規劃_v3_FlashVSR_Integrated.md`

---

## 現況摘要

| 項目 | 現況 |
|------|------|
| Git Log 最新 | W8A8 SmoothQuant → **9.07 dB PSNR** vs FP16 baseline |
| 問題根因 | Activation 量化誤差傳播嚴重；observer max 收集被 outlier 支配 |
| WanModel 架構 | dim=1536, 12 heads, 30 layers, ffn_dim=6144 |
| 主要量化目標 | SelfAttention QKV/O + CrossAttention QKV/O + FFN |

---

## 發現的關鍵問題

### 1. SmoothQuant forward 精度問題
`SmoothQuantLinear.forward()` 將 `x` 量化成 int8 再反量化回 fp32 做 matmul：
```python
x_int8 = torch.round(x / self.act_scale).to(torch.int8)
y = F.linear(x_int8.float(), w_fp16.float(), self.bias.float())
```
兩個 fp32 intermediate 乘積可能累積誤差，且 activation scale 不夠精準時影響更大。

### 2. ObserverLinear 收集方式不夠魯棒
使用 `torch.max` 而非 percentile/MSE-optimal：單一 outlier 就會讓整層 scale 偏大。

### 3. 沒有區分 attention 層 vs FFN 層
兩種層的 activation 特性差異很大，應該差異化處理。

---

## PR 規劃（5+1個 PR）

| PR | 名稱 | 優先級 | 關鍵技術 |
|----|------|--------|---------|
| PR #0 | 診斷與基準建立 | **緊急** | Per-layer ablation |
| PR #1 | Temporal-Aware Calibration | 高 | QuantVSR STCA + PMQ-VE |
| PR #2 | CSB for WanModel Attention | 高 | PTQ4DiT CSB + per-head QKV split |
| PR #3 | FFN INT8 + 殘差處理 | 中 | FFN per-layer calibration |
| PR #4 | A16 策略研究 | 中 | 哪些模塊用 A16 |
| PR #5 | 端到端整合驗證 | 高 | 完整 benchmark |

---

## 目標

- PSNR > 30 dB vs FP16 baseline（目前 9.07 dB，严重不足）
- VRAM 節省 > 30%

## 待確認

1. 測試用的 reference video 是哪支？（PR #0 需要盡快跑 per-layer ablation）
2. FP16 baseline 的 PSNR 是多少？（還沒有建立）
3. 最終 A8W8 目標是 pure INT8 還是允許部分 A16？
