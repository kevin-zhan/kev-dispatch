# 合成派单样本规格（Kev 训练用）

每条样本 = AIO Agent 派单时发给决策模型的一次请求 + 正确答案。必须和生产格式逐字一致（模板是英文，内容是中文）。
参考真实样本的形状（只看格式，**不要抄内容、人名、地址、订单号**）：`/Users/mengxiao/workspace/.scratch/tmp/kev-dispatch/data/development.jsonl` 前 3 行。

## 虚构的使用者

一个住在美国湾区、说中文的家庭用户（有婴儿、养猫、开车、网购、带娃出游、关注优惠、偶尔问科技/投资/健康/签证/菜谱/学习问题），用 AIO Agent 做：查资料、比价下单、订餐、规划行程、定时任务（每日推送、比价）、操作浏览器/手机、查监控摄像头、管理日程和邮件、写文档。
人名、商品、地点全部虚构或用常见公共品牌，不要用真实数据里的人名和私人细节。任务 ID 用 `task_` + 24 位随机小写十六进制。

## 一条样本（JSONL 一行）

```json
{"state": {"now": "2026-10-05T04:12:33.120Z",
           "main_session_timeline": "<时间线，见下>",
           "message": "<本条用户消息>"},
 "questions": {"route": {"type": "choice", "instructions": <INSTRUCTIONS 原样>, "criteria": {<选项>}, "label": "<正确选项键>"}},
 "_meta": {"synthetic": true, "scenario": "<场景代号>", "rationale": "<一句话说明为什么是这个答案>"}}
```

`instructions` 必须原样复制真实样本里的 `questions.route.instructions`（一个含 goal 和 rules 的对象）。

### 时间线 `main_session_timeline`
最近 6–10 条用户消息，按时间从旧到新，每行两个空格开头，最后一行是本条消息：
```
  [21:42] User: "牛肉和火鸡的含铁量一样吗" → task task_8bb5... "比较牛肉与火鸡肉的含铁量" (completed)
  [21:49] User: "寄出了，知识库里没有记录吗" → supplement to task task_c5cb... "核查快递是否寄出" (completed)
  [22:35] Scheduled run: "每天早上汇总三家促销网站的好价…" → task task_6cc7... "每日好价推送" (completed)
  [22:40] User: "帮我订周六晚上的餐厅" → task task_aa01... "预订周六晚餐餐厅" (needs_input); the assistant last asked: "要几个人、几点？"
▶ [22:41] User: "4个人，7点"  ← this message
```
- 时间 `HH:MM`（与 now 同一天）；跨天的写 `YYYY-MM-DD HH:MM`。
- 一个任务可以对应多条消息（后续消息续接同一个任务时，`task <同一个 id>`；合并进去的补充写 `supplement to task`）。
- 状态：completed / failed / running / queued / needs_input / interrupted。
- 只有任务最后一轮在等用户回答时才带 `; the assistant last asked: "..."`。

### 选项 `criteria`
- 时间线里出现的每个不同任务一个选项，按任务创建时间**从新到旧**编号：`1st most recent`、`2nd most recent`、`3rd`…（序数英文：1st 2nd 3rd 4th…11th 12th 13th 21st…）。
- 另外 0–3 个"按内容召回的较早任务"（不在时间线里，日期更早），放在最后。
- 键：任务状态为 running / queued / planning / waiting → `steer:<id>`；completed / failed / interrupted，或 needs_input → `resume:<id>`。
- 最后一个键固定是 `"NEW": "Handle as a new task; it may draw on an earlier task's background or results, but this time it has its own goal and deliverable"`。
- 描述格式（用 `; ` 连接，缺的部分省略）：
  `1st most recent (22:40) "预订周六晚餐餐厅", status needs_input; User's words: 帮我订周六晚上的餐厅; The assistant last asked the user: 要几个人、几点？; Result summary: …; Latest assistant message: …`
  召回任务开头是 `Earlier task recalled by content (2026-09-28 18:05) "标题", status completed`。
  User's words 和 Result summary 各不超过 200 字，超过就截断并以 `…` 结尾；Result summary 写得像真的助理回复（加粗结论、价格、链接、列表都可以）。running/queued 的任务一般没有 Result summary。

## 答案规则（与生产派单一致）
- 追问、指代、修改、确认、对上一个结果继续操作（"那后天呢""换成便宜点的""第二个""就这个，下单吧""改成周日"）→ resume 那个任务（运行中的 → steer）。
- 简短确认回应**最近一次向用户提问**的任务——那个任务不一定是最近的一条。
- 点名某个较早任务里的实体（人名、商品、地点、网站、文件）→ 那个任务，即使不是最近的（包括召回任务）。
- 只是关键词重合、或参考旧结果但有独立新目标和交付物 → NEW。
- 完全无关的新请求 → NEW。
- 用户在任务还在运行时补充要求/纠正 → steer 那个任务。

## 场景代号（每条样本选一个，写在 `_meta.scenario`）
- `confirm-latest`：简短确认/选择，回应最近一条在等用户的任务。
- `confirm-older`：简短确认/回复，但在等用户的是第 2–4 近的任务（更近的任务都已完成且没提问）。
- `followup-latest`：对最近任务结果的追问/修改。
- `followup-older`：追问的是第 2–5 近的任务（靠内容能看出来）。
- `named-recalled`：点名召回任务（或时间线里较早任务）中的实体。
- `steer-running`：给运行中/排队中的任务补充要求或改条件。
- `new-overlap`：和某个旧任务关键词/话题重合，但其实是独立新请求（应 NEW）。
- `new-builds-on`：借鉴旧结果但有独立新目标（应 NEW）。
- `new-unrelated`：完全无关的新请求（应 NEW）。
- `new-short`：很短但是独立的新请求（如"今天天气""讲个笑话"）（应 NEW）。

难度要真实：干扰项要像样（相近话题的其他任务、同样在等用户的另一个任务等），不要让答案一眼可见。
