# FlashVSR PTQ W8A8 TRT Compile — 進度紀錄

**日期:** 2026-05-22
**主題:** torch_tensorrt.dynamo INT8 量化流程重建

---

## 1. 問題確認

### torch_tensorrt.dynamo 不支援 INT8 autocast

原始 `compile_trt_w8a8.py` 使用：
```python
trt_model = dynamo.compile(
    model,
    inputs=[input_spec],
    enabled_precisions={torch.float32},
    use_explicit_typing=True,
    require_full_compilation=True,
    enable_autocast=True,
    autocast_low_precision_type=torch.int8,   # ← ERROR: 只支援 FP16/BF16
    autocast_calibration_dataloader=calibration_loader,
)
```

**根本原因：** 原始碼確認 `autocast_low_precision_type` 只接受 `torch.float16` / `torch.bfloat16`，傳 `torch.int8` 會觸發 validation error。

---

## 2. 解決方案：方向 A

**核心思路：** torch_tensorrt 只做圖擷取 + TRT engine 生成，不做量化。模型本身已經是 INT8 了，不需要 TRT 再量化一次。

**流程：**
```
Step 1: run_calibration()        → 收集 activation stats
Step 2: convert_model_to_w8a8() → 把 nn.Linear 換成 Int8ActLinear (W8A8)
Step 3: dynamo.compile()         → 只做圖擟取 + TRT engine 生成（不開 autocast）
```

**關鍵改動：**
```python
# dynamo.compile 調用（無 autocast）：
trt_model = dynamo.compile(
    model,
    inputs=[input_spec],
    enabled_precisions={torch.float16},  # 沒有 int8
    require_full_compilation=True,
    # 無 enable_autocast / autocast_low_precision_type
)
```

---

## 3. 已修改的檔案

**`scripts/ptq/compile_trt_w8a8.py`**

| 區塊 | 修改內容 |
|------|---------|
| `compile_trt_engine()` | 移除 `calibration_loader` / `num_samples` 參數；移除 `enable_autocast` / `autocast_low_precision_type`；拿掉內部 flash_attention patch（已上移到 main）|
| `main()` | 加入 Step 0 flash_attention patch（try/finally 保護）；新呼叫 `run_calibration()` → `convert_model_to_w8a8()` → `compile_trt_engine()` |
| docstring | 更新流程描述 |

---

## 4. 補充發現

- `run_calibration()` 內部已有 hook 機制（`ActivationCollector`），可收集所有 `nn.Linear` / `nn.Conv2d` / `nn.Conv3d` 的 activation min/max 統計
- `convert_model_to_w8a8()` 可接受 `act_stats` dict 來做 per-channel activation scale，格式為 `{layer_name: {'act_min', 'act_max', 'act_scale', 'zero_point'}}`
- `flash_attention` patch 需覆蓋 calibration 和 dynamo.compile 兩個階段，因為兩者都會做 forward pass

---

## 5. 待驗證

- [ ] 實際跑一次 `python scripts/ptq/compile_trt_w8a8.py --input_ckpt <path> --output_engine <path> --num_samples 320`
- [ ] 確認 dynamo.compile 不會錯誤
- [ ] 確認输出的 TRT engine 可被 pipeline 正確載入（nodes.py `W8A8_PTQ` 模式）

---

## 6. 用戶說明

- **方向 A 確認：** 用户选了方向 A（手動量化 → dynamo.compile）
- **CUDA 版本問題：** 用户有提到要 rollback CUDA 到 12，但還沒有執行。這可能影響 torch_tensorrt 能否正常運作（torch_tensorrt 2.12.0 目前抱怨 CUDA 13 不支援 TRT-LLM plugins）