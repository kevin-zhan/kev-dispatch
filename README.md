# kev-dispatch

专门做**任务上下文判断**的 Kev：AIO Agent 收到一条主会话消息时，读最近的时间线和候选任务，判断它应该
新开任务（`NEW`）、追加给仍在运行的任务（`steer:<id>`），还是在某个已结束任务的同一执行会话里续接（`resume:<id>`）。

模型是 [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b)（Qwen3.5-4B-Base + LoRA + 指针头，TypeSafe System One 同款接口）
在 owner 真实派单记录上的微调，问题文本与线上 `src/control/tasks/context.ts::routingQuestion()` 的 `route` 问题逐字一致，
可直接作为 Jev 的本地替代来回答 route 问题。

> 私有仓库：权重由含私人对话的数据训练而来，不要公开或上传 Hugging Face。训练数据不在仓库里。

## 内容

| 路径 | 说明 |
|---|---|
| `model/` | 选定的 r6 检查点：`adapter_model.safetensors`（LoRA r=16）、`head.pt`（指针头，含拟合温度 T=2.378）、tokenizer、训练配置与指标（Git LFS） |
| `results/` | 各轮实验的评测汇总（只有指标，没有对话内容） |
| `scripts/` | 数据导出、标签复核合并、合成数据规格与校验、Colab 微调评测脚本 |

## 用法

需要 GPU（或 32 GB 以上的 Apple Silicon）；用 [jaredpalmer/kev](https://github.com/jaredpalmer/kev) 的服务端：

```bash
git lfs pull
uv run --extra serve python -m kev.serve --run /path/to/kev-dispatch/model --port 8009
```

请求与 Jev 相同（`POST /v1/systemone`），state 为 `{now, main_session_timeline, message}`，问题 `route` 的
`instructions` 与 `criteria` 照 `routingQuestion()` 生成。检查点自带拟合好的温度，返回的概率已校准。
训练时状态+问题最长约 5.8k token（`--max_state 6144`），线上派单请求都在这个范围内。

## 数据

- **真实数据**：personal-agent 控制面 2026-10-01 至 10-14 的 owner 派单记录 1064 条（`recall_events` 的 Jev 轨迹 + 派单器最终决定）。
  10-12 之前的轨迹是中文模板，全部按 a7f37ce 的对照逐字转成现在的英文模板（含任务原话里拼入的
  `This task's earlier question:` / `User supplement:`）。
- **按时间切分**：训练 642（10/01–10/09）、校准 100（从训练期随机抽）、开发 130（10/09–10/10）、
  测试 192（10/10 10:01 之后，此前任何调优都没看过）。
- **标签复核**：派单器与 Jev 不一致的 202 条交给独立评审盲评（不告知哪个答案来自谁），中/高置信度的改判 74 条；
  低置信度保留派单器原标签。评测同时报告"复核标签"和"派单器原标签"两种准确率。
- **合成数据**：按 `scripts/synth_spec.md` 生成 400 条虚构家庭的派单样本（不含真实内容），重点补少见情形：
  回应较早任务的提问、点名召回任务、运行中任务的追加（steer）、关键词重合但其实是新任务。只进训练集。

## 训练

Colab A100 80GB，`scripts/kevft.py`（`kev_modal.py::run_train` 去掉 Modal）：从 `jaredpalmer/kev-4b` 热启动 LoRA 与指针头，
`--lr 5e-5 --epochs 2 --batch 1 --accum 8 --dtype bf16 --max_state 6144`，混入 1000 条 Kev 公开训练集 `decision-v7` 防遗忘；
在校准集上按最小 NLL 拟合温度；评测走服务路径（bf16、LoRA 合并）。单独训练约 45–60 分钟，评测约 8 分钟。

## 结果

开发集 130 条（选模型只看开发集）：

| 模型 | 复核标签 | 派单器标签 | Brier | 5% 错误率下可自动处理 | 公开 decision-v7（防遗忘） |
|---|---|---|---|---|---|
| Kev-4B 原版 | 36.1% | 34.6% | 0.778 | 0% | 86.0% |
| r1 真实+合成，lr 2e-5 | 84.6% | 82.3% | 0.196 | 79% | 85.6% |
| r2 仅真实，lr 2e-5 | 83.9% | 82.3% | 0.232 | 68% | 85.1% |
| r3 真实+合成，lr 5e-5 | 90.0% | 89.2% | 0.174 | 78% | 86.5% |
| r4 同 r3，3 轮 | 89.2% | 87.7% | 0.173 | 81% | 85.8% |
| r5 同 r3，换种子 | 89.2% | 89.2% | 0.181 | 72% | 86.5% |
| **r6 同 r3，修正后数据（选定）** | **89.2%** | **87.7%** | 0.183 | 77% | 84.9% |
| r7 同 r6，换种子 | 86.2% | 84.6% | 0.197 | 70% | 85.8% |
| Jev（线上，同一批记录） | 87.7% | 85.4% | 0.185 | 83% | — |

测试集 192 条（r6 只读一次）：

| | Kev r6 | Jev |
|---|---|---|
| 全部，复核标签 | 84.9% | 89.6% |
| 全部，派单器标签 | 82.3% | 86.5% |
| 中文模板时期（10/10–10/12，92 条） | 87.0% | 87.0% |
| 英文模板时期（10/12 之后，100 条） | 83.0% | 92.0% |
| 高置信（p≥0.9）却错 | 1.6% | 1.0% |

**结论**：在训练数据覆盖的时期，微调后的 Kev-4B 与 Jev 持平（开发集略高，测试集前半段相同）；
在最新流量（10/12 之后）上落后约 9 个点，主要错在把新任务判成续接（10/12 后新任务占 63%，还有摄像头、手机等新能力）。
这是时间漂移，不是格式问题：补译 `This task's earlier question` / `User supplement` 前后（r3→r6），这一段都是 83%。
暂时**不建议**替换线上 Jev；更多 10/12 之后的派单积累起来后，用同一套脚本重训。

另外：r3 的测试集分数在定位上述格式问题时看过一次（84.4% / 81.8%），之后没有再依据测试集调参。

## 复现

1. 控制面库只读快照 → `python3 scripts/export_kev.py --db SNAPSHOT.sqlite --out DATA`
2. 分歧样本盲评后 `python3 scripts/merge_verdicts.py DIR`；合成数据按 `synth_spec.md` 生成并用 `check_synth.py` 校验
3. Colab：`uv sync --frozen` 建 kev 独立环境 + `flash-linear-attention==0.5.2`，然后
   `python kevft.py baseline|train|score ...`

不在本机训练或运行 Kev；数据只留在 workspace `.scratch/` 和 Colab 运行时。
