# mini-swe-agent 插桩方案说明

## 1. 目标

这套插桩方案面向 **mini-swe-agent 在 SWE-bench 场景下的 workflow 分析**，重点解决下面几个问题：

- 一个实例从开始到结束，完整 trace 如何组织。
- agent 的 `run / step / query / execute_actions` 如何在 Phoenix 中可见。
- LLM 调用如何与 agent step 对齐。
- 工具调用如何被可靠捕捉。
- Docker sandbox 的启动与清理如何纳入同一条 trace。
- 单实例与批量运行时，trace 结构如何保持一致。

当前方案默认与 **LiteLLM 的 OpenInference 自动插桩**配合使用。

---

## 2. 总体设计

方案采用 **分层插桩**：

1. **启动层**：负责创建 `trace_provider`、注册 exporter、开启自动插桩。
2. **benchmark 层**：为每个 SWE-bench 实例创建一个最外层的总控 span。
3. **agent 层**：记录 `agent.run`、`agent.step`、`agent.query`、`agent.execute_actions`。
4. **tool 层**：在 `Environment.execute()` 中记录真正的工具执行 span。
5. **sandbox 层**：记录 Docker 容器的启动和清理。
6. **LLM 层**：由 LiteLLM 自动插桩生成 LLM spans。

核心原则：

- **不依赖全局 `subprocess` patch 作为主链路**。
- **以 `Environment.execute()` 作为工具调用的稳定边界**。
- **以 `benchmark.instance_run` 作为单个实例的根 span**，把环境创建、agent 执行、cleanup 收到同一条 trace 里。

---

## 3. trace 层级结构

一个单实例的理想 trace 结构如下：

```text
benchmark.instance_run                [AGENT]
├── benchmark.get_environment         [CHAIN]
│   └── sandbox.start                 [CHAIN]
├── agent.run                         [AGENT]
│   ├── agent.step                    [CHAIN]
│   │   ├── agent.query               [CHAIN]
│   │   │   └── litellm.completion    [LLM]
│   │   └── agent.execute_actions     [CHAIN]
│   │       ├── tool.exec             [TOOL]
│   │       └── tool.exec             [TOOL]
│   └── agent.step                    [CHAIN]
│       └── ...
└── sandbox.cleanup                   [CHAIN]
```

说明：

- `benchmark.instance_run` 是一个实例级别的总控 span。
- `agent.run` 是 agent 内部运行主流程。
- `agent.step` 对应一轮 agent 迭代。
- `agent.query` 包住模型请求，真正的 LLM span 由 LiteLLM 自动生成。
- `agent.execute_actions` 负责把一个 step 内的 action 分发到环境层。
- `tool.exec` 在环境层产生，是真正的工具执行边界。
- `sandbox.start / sandbox.cleanup` 让 Docker 生命周期进入同一条 trace。

---

## 4. 修改的文件

当前插桩方案涉及以下文件：

### 4.1 Agent 相关

- `src/minisweagent/agents/default.py`
- `src/minisweagent/agents/interactive.py`

作用：

- 为 agent 主流程增加语义 span。
- 让默认 `interactive` 运行链也能完整打点。
- 把 `step_index`、`action_index` 等元数据注入 action，供环境层继续使用。

### 4.2 Environment 相关

- `src/minisweagent/environments/local.py`
- `src/minisweagent/environments/docker.py`

作用：

- 在 `execute()` 中记录工具调用。
- Docker 环境同时记录：
  - 用户原始命令
  - Docker 外层 wrapper 命令
- 为 `_start_container()` 和 `cleanup()` 记录 sandbox 生命周期。

### 4.3 Benchmark 相关

- `src/minisweagent/run/benchmarks/swebench.py`
- `src/minisweagent/run/benchmarks/swebench_single.py`

作用：

- 增加 `benchmark.instance_run` 根 span。
- 确保 `get_sb_environment() -> agent.run() -> env.cleanup()` 在同一条 trace 内。
- 让单实例与批量运行都遵循同一套 trace 组织方式。

### 4.4 工具函数

- `src/minisweagent/utils/tracing.py`

作用：

- 提供统一的 tracing 辅助函数。
- 包括：
  - `set_openinference_kind`
  - `set_span_attributes`
  - `mark_span_ok`
  - `mark_span_error`
  - `preview_text`
  - `preview_json`
  - `normalize_action`
  - `get_action_command`
  - `get_trace_meta`

---

## 5. 关键设计点

## 5.1 为什么主插桩点选 `Environment.execute()`

mini-swe-agent 的工具调用流程是：

- agent 从模型输出中解析出 actions
- `execute_actions()` 遍历这些 actions
- 对每个 action 调用 `self.env.execute(action)`

因此：

- `DefaultAgent.execute_actions()` / `InteractiveAgent.execute_actions()` 是 **工具分发边界**
- `Environment.execute()` 是 **真正工具落地边界**

这比全局 patch `subprocess.run` 更稳定，也更接近 agent 语义。

---

## 5.2 为什么还要改 `interactive.py`

SWE-bench 默认运行链并不是只用 `DefaultAgent`。

`swebench_single` 和 `swebench` 默认都会通过 `get_agent(..., default_type="interactive")` 使用 `InteractiveAgent`。

而 `InteractiveAgent` 自己重写了：

- `query()`
- `step()`
- `execute_actions()`

如果只改 `default.py`，那么默认运行链会漏掉这三层 agent 语义 span。所以 `interactive.py` 必须一起改。

---

## 5.3 为什么 `sandbox.start` 和 `agent.run` 原来会分成不同 trace

如果没有实例级别的外层 span，调用顺序通常是：

```text
env = get_sb_environment(...)
agent.run(...)
env.cleanup()
```

这意味着：

- `sandbox.start` 发生在 `agent.run` 之前
- `sandbox.cleanup` 发生在 `agent.run` 之后
- 它们创建 span 时没有共同的 active parent span

结果就是它们会变成不同 root trace。

现在通过在 `swebench_single.py` 和 `swebench.py` 中增加 `benchmark.instance_run`，这个问题已经解决。

---

## 5.4 为什么外部 Python 启动器不一定要手写总控 span

如果 benchmark 内部已经有 `benchmark.instance_run`，那么外部启动器的主要职责就变成：

- 调用 `phoenix.otel.register(...)`
- 开启自动插桩
- 把 trace 发往 Phoenix
- 启动 `swebench_single` 或 `swebench`

也就是说：

- **外部启动器主要负责 tracing 基础设施初始化**
- **真正有分析价值的 workflow span 主要在 agent / env / benchmark 内部产生**

所以当前推荐做法是：

- 启动器里不再额外包一层 `run-swebench-instance`
- 根 span 统一放在 mini-swe-agent 内部的 `benchmark.instance_run`

这样结构更清晰，也更一致。

---

## 6. Span 命名与语义

当前主要 span 如下：

| Span 名称                   | Kind    | 作用                        |
| --------------------------- | ------- | --------------------------- |
| `benchmark.instance_run`    | `AGENT` | 单个实例的总控 span         |
| `benchmark.get_environment` | `CHAIN` | 创建并初始化 benchmark 环境 |
| `sandbox.start`             | `CHAIN` | 启动 Docker 容器            |
| `agent.run`                 | `AGENT` | agent 运行主循环            |
| `agent.step`                | `CHAIN` | 单次 step                   |
| `agent.query`               | `CHAIN` | step 内的模型查询阶段       |
| `agent.execute_actions`     | `CHAIN` | step 内 action 分发         |
| `tool.exec`                 | `TOOL`  | 真实工具执行                |
| `sandbox.cleanup`           | `CHAIN` | 清理 Docker 容器            |

OpenInference kind 是通过 `openinference.span.kind` 属性设置的。

---

## 7. 记录的关键信息

不同层级会记录不同属性。重点包括：

### 7.1 benchmark / agent 层

- `mini.instance_id`
- `mini.subset`
- `mini.split`
- `mini.step_index`
- `mini.messages_before`
- `mini.messages_after`
- `llm.cost.total`
- `input.value`
- `output.value`

### 7.2 tool 层

- `mini.step_index`
- `mini.action_index`
- `mini.tool.backend`
- `mini.tool.command`
- `mini.tool.command.user`
- `mini.tool.command.wrapper`
- `mini.tool.cwd`
- `mini.tool.timeout`
- `mini.tool.returncode`
- `output.value`

### 7.3 sandbox 层

- `mini.container.id`
- `mini.container.image`
- `mini.container.startup_command`

属性值较长时会自动截断，避免把超长 stdout/stderr 直接塞进 trace。

---

## 8. 运行方式

## 8.1 前提

需要：

- Phoenix 可访问
- LiteLLM 自动插桩已开启
- 你修改后的 mini-swe-agent 文件已覆盖到仓库
- 本地数据集路径可用，例如：

```bash
~/agent-stack/benchmarks/SWE-bench_Verified
```

---

## 8.2 单实例运行

推荐使用最简 Python 启动器，只做 tracing 初始化，然后调用 `swebench_single`。

例如 shell wrapper：

```bash
#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-sympy__sympy-15599}"

source ~/agent-stack/scripts/load_env.sh
source ~/agent-stack/.venv-agent/bin/activate

export OTEL_SERVICE_NAME=mini-swe-agent
export PHOENIX_PROJECT_NAME=mini-swe-agent
export PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:6006

exec python ~/agent-stack/scripts/swebench_trace_runner.py single "$INSTANCE_ID"
```

---

## 8.3 批量运行

同样通过一个最简 wrapper 启动：

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKERS="${1:-1}"

source ~/agent-stack/scripts/load_env.sh
source ~/agent-stack/.venv-agent/bin/activate

export OTEL_SERVICE_NAME=mini-swe-agent
export PHOENIX_PROJECT_NAME=mini-swe-agent
export PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:6006

exec python ~/agent-stack/scripts/swebench_trace_runner.py batch "$WORKERS"
```

---

## 9. 在 Phoenix 中应该看到什么

一个单实例 trace 中，通常应该能看到：

1. `benchmark.instance_run`
2. `benchmark.get_environment`
3. `sandbox.start`
4. `agent.run`
5. 多个 `agent.step`
6. `agent.query`
7. LiteLLM 自动生成的 `LLM` span
8. `agent.execute_actions`
9. 一个或多个 `tool.exec`
10. `sandbox.cleanup`

如果这几个 span 出现，但不在同一条 trace 里，通常说明：

- benchmark 根 span 没生效
- 你运行的不是修改后的 `swebench_single.py` / `swebench.py`
- 或者外部运行路径没有走到你覆盖后的仓库代码

---

## 10. 常见问题

## 10.1 为什么工具调用要在 environment 层打点，而不是 model 层

因为 model 只负责生成 action，真正把 action 执行为命令的是 environment。

因此 environment 层更接近真实工具执行边界。

---

## 10.2 为什么不把全局 `subprocess.run` patch 当主方案

因为它虽然能抓到底层命令，但会丢掉很多 agent 语义信息，比如：

- 这是第几个 step
- 这是第几个 action
- 这是 agent 原始命令还是 Docker wrapper 命令
- 这条命令属于哪个 benchmark instance

当前方案优先保证 workflow 语义完整，再让 subprocess 级 patch 作为兜底或校验手段。

---

## 10.3 为什么还需要外部 `register(...)`

因为它负责：

- 创建并注册 tracer provider
- 配置 exporter
- 开启自动插桩

agent 内部虽然会创建 span，但前提是 tracing 基础设施已经初始化完成。

可以理解为：

- 外部启动器负责“搭 tracing 管道”
- mini-swe-agent 内部负责“产生 workflow span”

---

## 11. 后续扩展建议

当前方案已经适合做单实例和批量 workflow 分析。下一步如果要增强 CPU / GPU 资源视角，可以继续加：

1. **vLLM `/metrics` 对齐**
   - request waiting/running/swapped
   - TTFT
   - queue time
   - prefill/decode time
   - KV cache usage

2. **系统级 CPU 观测**
   - `pidstat`
   - `perf`
   - `docker stats`
   - eBPF

3. **subprocess 级兜底事件**
   - 用于补全未通过 `Environment.execute()` 进入的系统命令

这样就能把：

- benchmark
- agent
- tool
- sandbox
- LLM
- CPU/GPU 资源事件

放到同一个分析框架里。

---

## 12. 当前方案一句话总结

当前 mini-swe-agent 插桩方案的核心是：

> 以 `benchmark.instance_run` 作为单实例根 span，
> 以 `agent.run / step / query / execute_actions` 表达 agent workflow，
> 以 `Environment.execute()` 作为工具调用主边界，
> 以 `sandbox.start / cleanup` 补足 Docker 生命周期，
> 并与 LiteLLM 自动插桩生成的 LLM spans 拼接成完整 trace。

