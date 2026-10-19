"use strict";
const $ = (id) => document.getElementById(id);
let active = null;
let mode = "demo";
let savedWorkRecord = null;
let startedAt = 0;
let tick = null;
const urlToken = new URLSearchParams(location.search).get("key") || "";
const initialTask = new URLSearchParams(location.search).get("task") || "";
if (urlToken) sessionStorage.setItem("share_ui_key", urlToken);
const accessToken = urlToken || sessionStorage.getItem("share_ui_key") || "";
if (urlToken) history.replaceState(null, "", location.pathname);

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {headers: {"X-SHARE-TOKEN": accessToken}} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-SHARE-UI": "1", "X-SHARE-TOKEN": accessToken},
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败：${response.status}`);
  return result;
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

function showTimer() {
  const seconds = Math.floor((Date.now() - startedAt) / 1000);
  $("timer").textContent = `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`;
}

function text(id, value) { $(id).textContent = value == null ? "" : String(value); }
function pretty(value) { return JSON.stringify(value, null, 2); }

async function openTask(taskId, button) {
  if (active) { notice("请先提交当前任务或记录停止原因，再切换任务。", true); return; }
  try {
    const result = await api("/api/session", {task_id: taskId});
    active = result;
    document.querySelectorAll(".task-item").forEach(el => el.classList.toggle("active", el === button));
    const task = result.task;
    const visible = task.decision_time_information;
    const operation = visible.operation || {};
    text("task-kicker", `${task.failure_type} · ${task.complexity}`);
    text("task-title", operation.operation_name || "待修复任务");
    text("task-id", task.episode_id);
    text("complexity", task.complexity);
    text("task-description", visible.task || "未提供");
    text("operation", pretty(operation));
    text("failure", pretty(visible.observed_failure || {}));
    text("logs", (visible.available_logs || []).length ? pretty(visible.available_logs) : "无可用日志");
    $("candidate-section").classList.toggle("hidden", task.candidate_repair == null);
    if (task.candidate_repair != null) text("candidate", pretty(task.candidate_repair));
    text("fixture-hash", task.fixture_sha256);
    $("allowed-actions").replaceChildren(...(visible.allowed_actions || []).map(action => {
      const span = document.createElement("span"); span.className = "chip"; span.textContent = action; return span;
    }));
    $("repair-form").reset();
    $("target").value = operation.operation_name || "";
    $("bindings").value = "{}";
    $("saved").classList.add("hidden");
    $("revert-change").classList.add("hidden");
    $("revert-change").disabled = false;
    $("apply-change").disabled = false;
    $("stop-work").disabled = false;
    $("submit-repair").disabled = false;
    $("stop-task").disabled = false;
    savedWorkRecord = null;
    $("repair-form").classList.toggle("hidden", mode === "work");
    $("work-editor").classList.toggle("hidden", mode !== "work");
    if (mode === "work") {
      text("editable-path", task.editable_file.path);
      text("editable-format", `${task.editable_file.format.toUpperCase()} 源文件`);
      $("workflow-editor").value = task.editable_file.text;
      $("work-notes").value = "";
      $("preview-panel").classList.add("hidden");
    }
    $("empty").classList.add("hidden");
    $("workspace").classList.remove("hidden");
    startedAt = Date.now(); clearInterval(tick); showTimer(); tick = setInterval(showTimer, 1000);
    notice(mode === "work" ? "任务已打开。编辑右侧源文件，可先预览差异，再保存到真实工作目录。" : "任务已打开。计时已开始；请依据左侧可见材料填写方案。提交后只记录人工操作，不产生成功标签。");
  } catch (error) { notice(error.message, true); }
}

function repairArtifact() {
  let bindings;
  try { bindings = JSON.parse($("bindings").value); }
  catch { throw new Error("参数绑定必须是有效的 JSON 对象。"); }
  if (!bindings || Array.isArray(bindings) || typeof bindings !== "object") throw new Error("参数绑定必须是 JSON 对象。");
  return {
    action: $("action").value.trim(), target_operation: $("target").value.trim(),
    parameter_bindings: bindings,
    safety_checks: $("checks").value.split("\n").map(x => x.trim()).filter(Boolean),
    rationale: $("rationale").value.trim()
  };
}

async function submit(decision) {
  if (!active) return;
  const notes = $("notes").value.trim();
  if (decision === "participant_stop" && !notes) { notice("停止时请在操作备注中填写原因。", true); $("notes").focus(); return; }
  let payload = {session_id: active.session_id, decision, notes};
  try { if (decision === "submitted_repair") payload.repair_artifact = repairArtifact(); }
  catch (error) { notice(error.message, true); return; }
  $("submit-repair").disabled = true; $("stop-task").disabled = true;
  try {
    const result = await api("/api/submit", payload);
    clearInterval(tick);
    $("repair-form").classList.add("hidden");
    $("saved").classList.remove("hidden");
    text("saved-detail", result.completion_reason === "submitted_repair" ? "修复方案已留存，尚未执行或判定结果。" : "停止决定已留存，尚未判定结果。");
    text("record-id", `记录 ID：${result.record_id}`);
    $("revert-change").classList.add("hidden");
    notice("记录保存成功。此记录不计入修复成功率，直到受控执行和独立判定完成。");
    active = null;
  } catch (error) {
    notice(error.message, true);
    $("submit-repair").disabled = false; $("stop-task").disabled = false;
  }
}

async function previewWork() {
  if (!active) return;
  try {
    const result = await api("/api/preview", {session_id: active.session_id, updated_text: $("workflow-editor").value});
    text("diff-preview", result.changed ? result.diff : "文件内容没有变化。");
    $("preview-panel").classList.remove("hidden");
    notice(result.changed ? "文件语法有效，差异已生成。检查后可保存到文件。" : "文件没有变化；请修改后再保存。");
  } catch (error) { notice(error.message, true); }
}

async function submitWork(decision) {
  if (!active) return;
  const notes = $("work-notes").value.trim();
  if (decision === "participant_stop" && !notes) { notice("停止时请填写原因。", true); $("work-notes").focus(); return; }
  const payload = {session_id: active.session_id, decision, notes};
  if (decision === "apply_patch") payload.updated_text = $("workflow-editor").value;
  $("apply-change").disabled = true; $("stop-work").disabled = true;
  try {
    const result = await api("/api/submit", payload);
    clearInterval(tick);
    $("work-editor").classList.add("hidden");
    $("saved").classList.remove("hidden");
    const validation = result.validation || {};
    let message = decision === "apply_patch" ? "工作流文件已修改；原版、修订版和差异已留存。" : "停止决定已记录，工作流文件未改动。";
    if (validation.configured_check_ran) message += validation.passed ? " 预设校验通过。" : " 预设校验未通过，请查看保存记录。";
    else if (decision === "apply_patch") message += " 文件语法已通过，任务未配置额外校验。";
    text("saved-detail", message);
    text("record-id", `记录 ID：${result.record_id}`);
    savedWorkRecord = decision === "apply_patch" ? result.record_id : null;
    $("revert-change").classList.toggle("hidden", savedWorkRecord === null);
    notice("记录已保存。校验结果与人工修复成功判定分开记录。");
    active = null;
  } catch (error) {
    notice(error.message, true);
    $("apply-change").disabled = false; $("stop-work").disabled = false;
  }
}

async function revertWork() {
  if (!savedWorkRecord) return;
  $("revert-change").disabled = true;
  try {
    await api("/api/revert", {record_id: savedWorkRecord});
    text("saved-detail", "原文件已经恢复；撤销记录已单独留存。重新打开任务可继续修改。");
    notice("已恢复修改前的文件，并记录了此次撤销。");
    savedWorkRecord = null;
    $("revert-change").classList.add("hidden");
  } catch (error) { notice(error.message, true); $("revert-change").disabled = false; }
}

async function initialize() {
  try {
    const meta = await api("/api/meta");
    mode = meta.mode;
    const demo = meta.mode === "demo";
    text("mode-badge", demo ? "非研究演示" : mode === "work" ? "真实文件工作模式" : "已授权收集");
    $("mode-badge").classList.toggle("demo", demo);
    text("task-count", meta.tasks.length);
    notice(demo ? "当前是非研究演示模式，仅使用 dummy 任务；正式收集入口由现有九项门槛控制。" : mode === "work" ? "工作模式会写入配置目录中的真实源文件；每次保存都会留存原版与差异。" : "当前为正式任务材料。请仅使用允许的受控环境，不查看 AI 或控制器输出。");
    for (const task of meta.tasks) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "task-item";
      const title = document.createElement("strong"); title.textContent = `${String(task.order).padStart(3,"0")}  ${task.failure_type}`;
      const subtitle = document.createElement("span"); subtitle.textContent = `${task.complexity} · ${task.id.slice(-12)}`;
      button.append(title, subtitle); button.addEventListener("click", () => openTask(task.id, button));
      button.dataset.taskId = task.id;
      $("task-list").append(button);
    }
    if (initialTask) {
      const target = [...document.querySelectorAll(".task-item")].find(button => button.dataset.taskId === initialTask);
      if (target) target.click();
    }
  } catch (error) { notice(error.message, true); }
}

$("repair-form").addEventListener("submit", event => { event.preventDefault(); submit("submitted_repair"); });
$("stop-task").addEventListener("click", () => submit("participant_stop"));
$("preview-change").addEventListener("click", previewWork);
$("apply-change").addEventListener("click", () => submitWork("apply_patch"));
$("stop-work").addEventListener("click", () => submitWork("participant_stop"));
$("revert-change").addEventListener("click", revertWork);
initialize();
