# 项目状态文档

本目录保存可随代码一起备份的投稿进展材料。

- `电路图智能评价系统_当前进展与投稿要求.docx`：当前完成内容与投稿要求汇总。
- `build_publication_brief.py`：重新生成上述 DOCX 的脚本。

实验参考图和采集协议统一位于
`benchmark/blind_reference_pack/v1/`；答案键仅保留在本地完整工作树，模型权重、原始数据和本地缓存不纳入 GitHub 备份。

GitHub 安全备份只包含参考图，不包含 `answer_key/`、benchmark 的检测/GT 目录、OCR 标签、原始数据或本地 API 凭据。
