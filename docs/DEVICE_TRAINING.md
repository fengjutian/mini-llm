# 设备训练已统一

项目曾维护独立的 `DeviceTensor`、设备反向图和设备优化器。该实现现已移除，避免教学模型、自动微分和训练循环各维护两份。

现在只有一条训练链路：

```text
Tensor -> TensorValue -> TinyGpt -> Trainer
```

`mini-llm-backend` 只保留 kernel backend 接口。使用 `--backend candle` 时，仍然运行同一个 `TinyGpt` 与 `Trainer`，仅将矩阵乘法、Softmax 和 RMSNorm 等 kernel 路由到 Candle。

```bash
cargo run -p mini-llm-cli --features candle -- train --backend candle
```

这种结构优先保证算法透明、梯度路径唯一以及 CPU/Candle 结果可比较。
