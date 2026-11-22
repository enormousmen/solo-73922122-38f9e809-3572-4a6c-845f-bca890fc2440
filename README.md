# LoRA 小内存推理节点

一个用于演示 **小内存推理节点调度** 的 Web 服务：用户从预置 LoRA 适配器中选一个、提交输入向量，页面展示排队状态、内存预算占用和计算结果。

数值部分用 NumPy 在 CPU 上真正跑一个线性层 `y = (x·Aᵀ)·Bᵀ × scale + b`，并模拟加载与计算时延。调度规则按题目要求实现，见下。

## 快速启动

```bash
docker compose up --build
# 打开 http://localhost:8000
```

预算可通过环境变量调整（`docker-compose.yml` 中的 `NODE_MEMORY_BYTES`，默认 30000 字节）。

不使用 Docker 时：

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 页面功能

- **提交请求**：选择预置适配器、粘贴/生成 128 维向量（留空则由服务端生成确定性样例）、设置计算耗时、可勾选“模拟加载失败”。
- **内存预算**：已用/剩余字节与进度条；驻留权重表（适配器、精确字节数、引用计数、加载中/在用/空闲）。
- **计算位**：两个计算位的实时占用与进度。
- **等待队列**：位置、适配器、状态、已等待时间；队首缺内存时会标红并说明原因。
- **最近完成/结果**：点击任意行查看完整 96 维输出或错误信息。
- **事件时间线**：提交、加载开始/完成/失败、等内存、越过队首、派发、计算完成、卸载等事件。
- **运行样例**：一键跑脚本化的演示（见下文），**重置节点**清空队列、权重与演示状态。

## 调度规则与实现位置

| 规则 | 实现 |
| --- | --- |
| 同一适配器只保留一份权重，请求共享引用计数 | `Scheduler.resident`、`Resident.refcount` |
| 从开始加载起按数组字节数占用预算 | `_begin_load()` 先扣 `nbytes` 再异步加载 |
| 权重被在用请求引用时不可卸载；计算结束才释放引用 | `_compute()` 结束时 `refcount -= 1`；`_try_make_space()` 只挑 `refcount == 0 and not loading` |
| 最多两个计算位 | `MAX_CONCURRENT_COMPUTES = 2`，`_pump()` 检查 `len(self.running)` |
| 队首缺内存 + 有空闲计算位 → 放行一单使用已驻留权重的后到请求 | `_find_bypass_candidate()` + `head.skipped_head` |
| 每个队首最多被越过一次 | `Request.skipped_head`；越过一次后 `_pump()` 直接 `return` |
| 能卸载空闲权重腾出空间时优先启动队首 | 先 `_could_fit()`/`_try_make_space()`，只有腾不出空间才考虑绕过 |
| 加载失败只报错该请求并归还额度，运行中的计算不受影响 | `_load_failed()`：移除请求、`used_bytes -= nbytes`，不触碰 `running` |

“数组字节数”严格按 `np.float32` 元素数计算：`(rank×128 + 96×rank + 96) × 4`。预置适配器：

| key | 名称 | rank | 字节 |
| --- | --- | --- | --- |
| support | 支持语气 | 8 | 7 552 |
| finance | 财务口吻 | 16 | 14 720 |
| legal | 法律措辞 | 24 | 21 888 |
| brief | 极简摘要 | 32 | 29 056 |

默认 30 000 B 下，两个 `brief`（58 112 B）放不下，这正是绕过逻辑的触发条件。

## 脚本化样例（页面“运行样例”按钮 / `POST /api/demo`）

1. 提交 2 单 `legal`：只加载一份权重，两单共享，同时占满两个计算位。
2. 等两单都进入计算后提交 `support`，权重填满预算并等待计算位。
3. 待 `support` 开始计算后提交 `brief`（29 056 B）：预算已满且在用权重不能卸载，它成为**队首并等待内存**。
4. 随后提交一单 `legal`：它的权重已驻留，因此**越过队首先跑一次**。
5. 紧接着再提交一单 `brief`：该队首已经让路一次，此单**必须等队首启动后才派发**，随后与队首共享权重。

脚本按调度器状态推进（而非固定 sleep），慢机器上顺序也不会漂移。

页面可在事件时间线中直接看到 `waiting_memory → head_bypassed → bypass_dispatched → head_holds_line` 的顺序。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/presets` | 预置适配器、维度、预算、计算位数量 |
| GET | `/api/state` | 队列、计算位、驻留权重、预算、事件、样例状态 |
| GET | `/api/requests/{id}` | 单请求详情与完整结果 |
| POST | `/api/requests` | `{adapter, vector?, compute_ms?, simulate_load_failure?}` |
| POST | `/api/demo` | 启动脚本化样例；样例在跑或节点仍有排队/计算中的请求时返回 409 |
| POST | `/api/reset` | 取消后台任务并清空节点 |

`vector` 必须是 128 个有限数值；省略时按请求序号生成确定性样例向量。

## 测试

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

覆盖：权重共享、加载即计费、空闲缓存复用与卸载、加载失败隔离与额度归还、
队首等内存/单次越过/不再二次越过、能腾空间时优先队首、计算位上限、
样例时线、NumPy 数值与参考实现一致、API 输入校验与重置。

## 项目结构

```
app/adapters.py    预置适配器、数组字节数、NumPy 线性层
app/scheduler.py   内存预算、驻留权重、队列、绕过与失败处理
app/demo.py        脚本化样例（按状态推进）
app/main.py        FastAPI 路由与生命周期清理
static/            单页前端（轮询 /api/state）
tests/             调度器、样例、API 测试
```

## 已知限制

- 调度器是进程内单实例状态，只应跑一个 uvicorn worker；多副本需要外部协调。
- 权重是确定性随机生成的“仿真 LoRA”，并非真实训练产物；加载/计算时延为模拟值。
- 卸载使用简单的空闲权重 LRU（`last_used`），没有命中率统计或预取。
- 没有持久化、鉴权或多用户隔离；重置会清空所有状态。
