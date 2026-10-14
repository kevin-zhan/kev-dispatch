# kev-dispatch 项目规范

先读 workspace 根规范 `/Users/mengxiao/workspace/AGENTS.md`，本文件只补充本项目。

## 用途

Kev-4B（jaredpalmer/kev）在 AIO Agent 派单"任务上下文判断"上的微调：读主会话时间线 + 新消息，在 NEW / steer / resume 某个已有任务之间选一个。接口与 TypeSafe System One（Jev）相同，可直接替换 `src/control/jev.ts` 的 route 问题。

## 目录

- `scripts/export_kev.py`：从 personal-agent 控制面库快照导出派单记录，统一转成线上英文模板（a7f37ce 之后），按时间切分。
- `scripts/merge_verdicts.py`：把盲评复核结果合回数据，生成 `*.adj.jsonl`。
- `scripts/synth_spec.md` / `scripts/check_synth.py`：合成样本规格与格式校验。
- `scripts/kevft.py`：Colab 单卡上的微调、温度校准、评测（`kev_modal.py::run_train` 去掉 Modal）。
- `model/`：LoRA 适配器 + 指针头（含拟合温度），Git LFS。
- `results/`：评测汇总（只有指标与 task id，没有对话内容）。

## 约束

- **不在本机训练或运行 Kev**（用户要求）；训练与评测只在 Colab GPU 上做。
- **数据不进仓库**：派单记录含用户私人对话，训练、复核、合成数据都只留在 workspace `.scratch/tmp/kev-dispatch/`（600 权限）和 Colab 运行时。仓库只放代码、权重和指标。
- 仓库为私有；权重由私人数据训练而来，不要公开或上传 Hugging Face。
- 往 Colab 传数据必须经用户授权；临时隧道必须用隔离的空配置启动（`HOME=<tmp> cloudflared tunnel --config <empty.yml> --url ...`），否则会读到 `~/.cloudflared/config.yml` 并以正式隧道（roy/tsl.zymx.tech）的身份注册连接器。
