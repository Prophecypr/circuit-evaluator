# CRNN-v2 手绘电路数值 OCR：Kaggle 训练说明

## 你需要上传什么

只上传交付目录中的 `ocr_crnn_hand_v2_kaggle.zip`。包内已经包含训练/验证裁剪、清单、代码、配置和 Notebook；不包含 API Key、Kaggle Token、42 张最终测试图或它们的 GT。

42 张图的连线 GT 和数值 GT 可以在训练期间继续标注，但不要把这些文件添加到 Kaggle 数据集，也不要根据当前模型预测修改人工答案。

## Kaggle 操作

1. 在 Kaggle 创建一个私有 Dataset，上传 `ocr_crnn_hand_v2_kaggle.zip`。Kaggle 如果自动解压，Notebook 会直接找到配置；如果保留 ZIP，Notebook 会自动在工作目录解压。
2. 新建 Notebook，把 `notebooks/crnn_v2_kaggle.ipynb` 导入，或在上传的数据集中打开同名 Notebook。
3. 把刚才的私有 Dataset 添加为 Notebook Input。
4. 在 Notebook Settings 中启用 GPU。Internet 可以关闭，本包不需要联网和任何 API Key。
5. 点击 **Run All**。第一个代码单元会定位数据，第二个会检查 GPU 和所有文件哈希，之后训练最多 40 个 epoch，并在连续 7 个 epoch 无改进时提前停止。
6. 最后一个单元会生成 `/kaggle/working/ocr_crnn_hand_v2_results.zip`。从 Output 面板下载该文件并原样交回项目目录。

## 正常输出

下载包中至少应有：

- `best.pt`：按验证集 Normalized Exact Match 选择的最佳权重。
- `last.pt`：最后一个 epoch 的权重。
- `history.csv`：每个 epoch 的损失、Exact Match、CER、学习率和耗时。
- `metrics.json`：最佳模型的完整指标和单位分组结果。
- `errors.csv`：验证集错误案例。
- `run_config.json`：实际使用的配置。
- `environment.txt`：Python、PyTorch、OpenCV、GPU 和平台信息。
- `independent_eval/`：训练结束后重新加载 `best.pt` 得到的独立复算结果。

## 判定规则

模型不会因为训练完成就自动替换现有 OCR。下载结果后，要在相同的未见绘图者验证集上独立复算：

- Normalized Exact Match 不低于 80%；
- Normalized CER 不高于 10%；
- 同一验证清单上优于当前 `runs/ocr_crnn_machine/best.pt`。

三项均满足才备份并接入主流水线。达不到时只分析验证集 `errors.csv`，不能查看或利用 42 张最终测试图的 GT 调参。

## 常见问题

- 报错 `GPU is not enabled`：在 Notebook Settings 中选择 GPU 后重新 Run All。
- 报错 `checksum mismatch`：上传文件不完整或被修改，重新上传原始 ZIP。
- 报错找不到配置：确认把 `ocr_crnn_hand_v2_kaggle.zip` 作为 Input 添加到了当前 Notebook。
- 运行中断：Kaggle 会保留已产生的输出，但为保证可复现，应从头 Run All；不要手改数据清单或字符集。
- 显存不足：先把配置中的 `batch_size` 从 64 改为 32，并在实验记录中明确记录；其他配置不变。
