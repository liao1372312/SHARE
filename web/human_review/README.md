# SHARE 真人修复工作台

本地网页有三个模式：`work` 直接编辑配置目录中的真实文件；`demo` 使用非研究 dummy 任务；`collect` 按既有 Plan-B 门槛记录正式研究任务。所有模式都把成功与 unsafe 留给独立判定；文件语法或校验命令通过不等于修复成功。

## 真正修改项目文件

项目已配置一个真实的待修问题：`flare/detection.py` 未识别复数键 `missing_parameters`。从项目根目录启动：

```bash
python -m share.human_review_ui --mode work --workspace-root . --tasks web/human_review/project_tasks.json --operator-id reviewer01 --port 8765
```

打开启动时打印的带访问 token 的本地链接。选择任务、编辑右侧 Python 文件、预览差异，再点击“保存到文件”。保存会直接修改 `flare/detection.py`，同时在 `results/share_human_review_ui/work/<record_id>/` 留存原版、修改版、差异、时间、哈希和校验输出。保存后可用界面的“撤销本次文件修改”恢复原版；如果文件已被再次编辑，自动撤销会拒绝覆盖。这个任务的预设校验为 `python -m scripts.check_detector_plural_parameter_key`；当前原始代码会使它失败，直到真人完成修复。保存前会拒绝语法错误和已被其他进程修改的文件。

接入自己的任务时，在工作目录放置 `.json`、`.py` 或 `.toml` 文件，再准备任务清单。例如：

```json
{
  "schema_version": 1,
  "tasks": [{
    "id": "my-workflow-01",
    "title": "修复工作流参数",
    "request": "根据可见日志修复参数绑定。",
    "workflow_file": "workflows/flow.json",
    "failure_type": "parameter_binding_failure",
    "observed_failure": {"message": "missing city"},
    "logs": [{"level": "error", "message": "city is required"}],
    "validation_command": ["python", "-m", "json.tool", "{file}"]
  }]
}
```

`workflow_file` 必须在 `--workspace-root` 内，不能引用外部路径。`validation_command` 是可选的参数数组，服务端不通过 shell 执行；`{file}` 会替换为被编辑文件的绝对路径。请只在自己信任的任务清单中配置命令。工作模式是日常修复工具，记录不自动进入论文的正式真人实验。

## 非研究演示

在项目根目录运行：

```bash
python -m share.human_review_ui --mode demo --port 8765
```

在浏览器打开 `http://127.0.0.1:8765/`。演示只加载现有的两个 `dry_run=true` dummy 任务。打开任务时会在 `results/share_human_review_ui/demo/session_starts/` 留存开始记录；提交写入 `results/share_human_review_ui/demo/<record_id>/`，其中包含 `submission.json`、可选的 `repair_artifact.json` 和 `SHA256SUMS`。它们不能计入论文的真人实验。

如果程序在远程 Linux 主机上运行，可通过 SSH 本地端口转发在自己的浏览器访问，例如 `ssh -L 8765:127.0.0.1:8765 user@host`。

## 正式任务入口

正式入口使用既有的 120 个冻结预决策 fixture，仅在最终九项收集门槛为 `READY_TO_COLLECT`、readiness 文件绑定同一审计哈希、且指定匿名操作员位于合格名单时启动：

```bash
python -m share.human_review_ui --mode collect --operator-id operator_<匿名编号> --port 8765
```

启动时打印的带一次性访问 token 的本地链接只应给该操作员。现有门槛状态是 `BLOCKED_DO_NOT_COLLECT`，所以正式入口当前会拒绝启动。演示模式不会读取这些冻结正式任务。

正式研究模式保存的是**操作员撰写的方案或停止决定**，并明确标记 `actual_repair_execution_verified=false`、`outcome_status=unadjudicated`。后续受控执行、工具日志、独立成功/unsafe 判定和冻结协议的分支封存仍需单独完成，不能把此目录直接并入论文成功率统计。

## 界面字段

- 可见：原始任务、目标操作、观察到的故障、现有日志、允许的操作和材料哈希。
- 可编辑：修复动作、目标操作、参数绑定 JSON、安全检查、修复依据、操作备注。
- 可选决定：提交修复方案；停止并填写原因。
- 服务端记录：匿名操作员、开始与完成时间、耗时、决定、材料哈希和修复方案哈希。不会保存用户自报的成功或安全标签。
