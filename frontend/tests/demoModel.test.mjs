import test from "node:test";
import assert from "node:assert/strict";
import {
  adaptRun,
  smoothCurve,
  executionSummary,
  modelFamily,
  modelLabel,
} from "../src/demoModel.ts";

test("the demo models get a family and a short label", () => {
  assert.equal(modelFamily("Qwen/Qwen3-Coder-30B-A3B-Instruct"), "coder");
  assert.equal(modelFamily("openai/gpt-oss-120b"), "reasoning");
  assert.equal(modelFamily("Qwen/Qwen3-30B-A3B-Instruct-2507"), "general");
  assert.equal(
    modelLabel("Qwen/Qwen3-Coder-30B-A3B-Instruct"),
    "Qwen3-Coder-30B",
  );
  assert.equal(modelLabel("Qwen/Qwen3-30B-A3B-Instruct-2507"), "Qwen3-30B");
  assert.equal(modelLabel("openai/gpt-oss-120b"), "gpt-oss-120b");
});

test("a model call names the replica and GPU the gateway routed it to", () => {
  const run = adaptRun({
    agent: {
      name: "A",
      goal: "",
      agents: [{ id: "task1", name: "Analyst", task: "" }],
    },
    run: {
      started: 0,
      finished: 3,
      status: "completed",
      output: "",
      events: [
        {
          id: "1",
          kind: "node_started",
          created: 0,
          payload: { task_id: "task1", task_name: "Analyst" },
        },
        {
          id: "2",
          kind: "model_request",
          created: 1,
          payload: {
            request_id: "r1",
            task_id: "task1",
            task_name: "Analyst",
            model: "general",
          },
        },
        {
          id: "3",
          kind: "model",
          created: 2,
          payload: {
            request_id: "r1",
            task_id: "task1",
            task_name: "Analyst",
            placement: {
              model_id: "Qwen/Qwen3-30B-A3B-Instruct-2507",
              device: "cuda",
              gpus: [3],
              container: "vllm-gpu-qwen-qwen3-30b-a3b-instruct-2507-2",
            },
          },
        },
        {
          id: "4",
          kind: "node_completed",
          created: 3,
          payload: { task_id: "task1", task_name: "Analyst" },
        },
      ],
    },
  });
  const call = run.tasks[0].calls[0];
  assert.equal(call.model, "Qwen/Qwen3-30B-A3B-Instruct-2507");
  assert.equal(call.where, "GPU 3");
  assert.equal(call.family, "general");
  assert.deepEqual(run.models, [{ family: "general", label: "Qwen3-30B" }]);
  assert.ok(
    run.activity.some((a) =>
      a.text.includes("Qwen3-30B-A3B-Instruct-2507 on GPU 3"),
    ),
  );
});

test("missing telemetry and unfinished calls are not presented as successful measurements", () => {
  const rec = {
    agent: { agents: [{ id: "a", name: "Worker", task: "Check data." }] },
    run: {
      started: 100,
      finished: 103,
      status: "failed",
      output: "",
      events: [
        {
          id: 1,
          created: 100,
          kind: "node_started",
          payload: { node_id: "a" },
        },
        {
          id: 2,
          created: 101,
          kind: "model_request",
          payload: { task_id: "a", request_id: "r" },
        },
        {
          id: 3,
          created: 102,
          kind: "resources",
          payload: { gpus: [{ utilization_percent: null }] },
        },
      ],
    },
  };
  const run = adaptRun(rec);
  assert.equal(run.tasks[0].calls[0].end, null);
  assert.equal(run.tasks[0].end, null);
  assert.equal(run.samples[0].cpu, null);
  assert.equal(run.samples[0].gpu, null);
  assert.match(run.activity.at(-1).text, /stopped/);
});

test("smooth curves retain measured values without overshooting any interval", () => {
  const points = [
    [0, 0],
    [2, 100],
    [5, 100],
    [9, 20],
    [10, 40],
    [12, 0],
  ];
  const curve = smoothCurve(points);
  for (const [x, y] of points) assert.ok(Math.abs(curve.value(x) - y) < 1e-8);
  for (let i = 0; i < points.length - 1; i++)
    for (let f = 0; f <= 1; f += 0.01) {
      const [x, y] = points[i],
        [nextX, nextY] = points[i + 1];
      const v = curve.value(x + f * (nextX - x));
      assert.ok(
        v >= Math.min(y, nextY) - 1e-8 && v <= Math.max(y, nextY) + 1e-8,
      );
    }
});

test("execution summary excludes future calls and readings and preserves zero utilization", () => {
  const run = {
    duration: 10,
    tasks: [
      {
        calls: [
          { kind: "model", start: 1 },
          { kind: "tool", start: 2 },
          { kind: "model", start: 8 },
        ],
      },
    ],
    samples: [
      { t: 0, cpu: 0, gpu: null },
      { t: 2, cpu: 20, gpu: 40 },
      { t: 8, cpu: 100, gpu: 100 },
    ],
  };
  assert.deepEqual(executionSummary(run, 0), {
    llmCalls: 0,
    cpuAverage: 0,
    gpuAverage: null,
  });
  assert.deepEqual(executionSummary(run, 2), {
    llmCalls: 1,
    cpuAverage: 10,
    gpuAverage: 40,
  });
  assert.deepEqual(executionSummary(run, 10), {
    llmCalls: 2,
    cpuAverage: 40,
    gpuAverage: 70,
  });
  assert.deepEqual(executionSummary({ ...run, samples: [] }, 2), {
    llmCalls: 1,
    cpuAverage: null,
    gpuAverage: null,
  });
});

test("the legend lists the models from the lightest shade to the darkest", () => {
  const call = (id, t, model) => [
    {
      id: id + "a",
      kind: "model_request",
      created: t,
      payload: { request_id: id, task_id: "task1", task_name: "Analyst" },
    },
    {
      id: id + "b",
      kind: "model",
      created: t + 1,
      payload: {
        request_id: id,
        task_id: "task1",
        task_name: "Analyst",
        placement: { model_id: model, device: "cuda", gpus: [0] },
      },
    },
  ];
  const run = adaptRun({
    agent: {
      name: "A",
      goal: "",
      agents: [{ id: "task1", name: "Analyst", task: "" }],
    },
    run: {
      started: 0,
      finished: 10,
      status: "completed",
      output: "",
      events: [
        {
          id: "s",
          kind: "node_started",
          created: 0,
          payload: { task_id: "task1", task_name: "Analyst" },
        },
        ...call("r1", 1, "openai/gpt-oss-120b"),
        ...call("r2", 3, "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        ...call("r3", 5, "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        {
          id: "e",
          kind: "node_completed",
          created: 10,
          payload: { task_id: "task1", task_name: "Analyst" },
        },
      ],
    },
  });
  assert.deepEqual(
    run.models.map((m) => m.family),
    ["coder", "general", "reasoning"],
  );
});
