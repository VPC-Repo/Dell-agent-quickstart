import test from "node:test";
import assert from "node:assert/strict";
import {
  adaptRun,
  smoothCurve,
  executionSummary,
  modelFamily,
  modelLabel,
  smoothSteps,
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

test("an agent's footprint comes from its own measured steps: cores, requests and the CPU : GPU ratio", () => {
  const ev = (id, created, kind, payload) => ({ id, created, kind, payload });
  const place = {
    gpus: [3],
    container: "vllm-a",
    model_id: "Qwen/Qwen3-30B-A3B-Instruct-2507",
  };
  const record = {
    agent: {
      id: "a",
      demo: { slug: "credit-risk" },
      agents: [{ id: "task1", name: "Modeler" }],
      goal: "g",
    },
    run: {
      started: 100,
      finished: 110,
      status: "completed",
      output: "",
      events: [
        ev(1, 100, "attempt", {}),
        // 100 tokens on GPU 3, then a 2 s sandbox job using 60 core-s, then 200 tokens
        ev(2, 101, "model_request", {
          request_id: "m1",
          task_id: "task1",
          placement: place,
        }),
        ev(3, 102, "model", {
          request_id: "m1",
          task_id: "task1",
          started_at: 101,
          timestamp: 102,
          output_tokens: 100,
          placement: place,
        }),
        ev(4, 103, "tool_started", {
          request_id: "t1",
          task_id: "task1",
          name: "simulate_credit_risk",
        }),
        ev(5, 105, "tool_completed", {
          request_id: "t1",
          task_id: "task1",
          work_evidence: {
            cpu_seconds: 60.5,
            caller_cpu_seconds: 0.5,
            sandbox_cpu_seconds: 60,
            units: { processes: 344 },
          },
        }),
        ev(6, 106, "model_request", {
          request_id: "m2",
          task_id: "task1",
          placement: place,
        }),
        ev(7, 108, "model", {
          request_id: "m2",
          task_id: "task1",
          started_at: 106,
          timestamp: 108,
          output_tokens: 200,
          placement: place,
        }),
        // the run process used 2.5 s of CPU, 0.5 of it the tool's calling thread; the services 0.5 s
        ev(8, 109, "cpu_segment", {
          core_ms: 2500,
          services: [
            { role: "warehouse", core_ms: 500 },
            { role: "model gateway", core_ms: 900 },
          ],
        }),
        ev(9, 110, "completed", {}),
      ],
    },
  };
  const run = adaptRun(record, { physicalCores: 172, sockets: 2 });
  const f = run.footprint;
  // 60 sandbox + 0.5 calling thread + 2.0 rest of the run process + 0.5 warehouse; the gateway is the model service's
  assert.equal(Math.round(f.coreSeconds * 10) / 10, 63);
  // during the job: 60.5 core-s over 2 s, plus 2.5 core-s spread over the 10 s run
  assert.equal(Math.round(f.peakCores * 100) / 100, 30.5);
  assert.equal(f.outputTokens, 300);
  assert.equal(run.engines.length, 1);
  assert.equal(run.engines[0].place, "GPU 3");
  assert.equal(Math.max(...run.engines[0].requests.map((s) => s.value)), 1);
  // 210 core-ms per token x 950 tokens/s per GPU / (1000 x 86 cores per socket), as deploy/cpu_per_token.py has it
  assert.equal(Math.round(f.socketsPerGpu * 100) / 100, 2.32);
  assert.equal(f.coresPerSocket, 86); // for the explanation: 210 x 950 / 1,000 = 200 cores, / 86 = 2.3 CPUs
});

test("a machine without GPUs has no CPU : GPU ratio", () => {
  const ev = (id, created, kind, payload) => ({ id, created, kind, payload });
  const place = {
    device: "cpu",
    container: "vllm-cpu",
    model_id: "Qwen/Qwen3-30B-A3B-Instruct-2507",
  };
  const run = adaptRun(
    {
      agent: {
        id: "a",
        demo: { slug: "faq-cache" },
        agents: [{ id: "task1", name: "Answerer" }],
        goal: "g",
      },
      run: {
        started: 100,
        finished: 110,
        status: "completed",
        output: "",
        events: [
          ev(1, 100, "attempt", {}),
          ev(2, 105, "model", {
            request_id: "m1",
            task_id: "task1",
            started_at: 101,
            timestamp: 105,
            output_tokens: 100,
            placement: place,
          }),
          ev(3, 109, "cpu_segment", {
            core_ms: 1000,
            services: [{ role: "model engine", core_ms: 700000 }],
          }),
          ev(4, 110, "completed", {}),
        ],
      },
    },
    { physicalCores: 172, sockets: 2 },
  );
  assert.equal(run.engines[0].place, "CPU engine");
  assert.equal(Math.round(run.footprint.coreSeconds), 1); // the CPU engine's own CPU is not the agent's
  assert.equal(run.footprint.socketsPerGpu, null);
});

test("smoothed curves keep the area of the steps they smooth", () => {
  // 50 cores for 2 s in a 10 s run: 100 core-seconds, drawn as a smooth bump
  const steps = [
    { t: 0, value: 0 },
    { t: 2, value: 50 },
    { t: 4, value: 0 },
    { t: 10, value: 0 },
  ];
  const curve = smoothSteps(steps, 10);
  const area = curve
    .slice(1)
    .reduce(
      (n, p, i) => n + ((p.value + curve[i].value) / 2) * (p.t - curve[i].t),
      0,
    );
  assert.ok(Math.abs(area - 100) < 0.5, `area ${area}`);
  assert.ok(Math.max(...curve.map((p) => p.value)) <= 50);
  assert.ok(curve[0].value < 0.01 && curve.at(-1).value < 0.01);
});

test("cores in use never read above the machine's cores", () => {
  const ev = (id, created, kind, payload) => ({ id, created, kind, payload });
  const run = adaptRun(
    {
      agent: {
        id: "a",
        demo: { slug: "full-machine" },
        agents: [{ id: "task1", name: "Analyst" }],
        goal: "g",
      },
      run: {
        started: 100,
        finished: 110,
        status: "completed",
        output: "",
        events: [
          ev(1, 100, "attempt", {}),
          // 600 core-s of job processes in 2 s: every hardware thread busy, more than the 172 cores
          ev(2, 102, "tool_started", {
            request_id: "t1",
            task_id: "task1",
            name: "reconcile_transactions",
          }),
          ev(3, 104, "tool_completed", {
            request_id: "t1",
            task_id: "task1",
            work_evidence: { cpu_seconds: 600, in_run_process: false },
          }),
          ev(4, 109, "cpu_segment", { core_ms: 0, services: [] }),
          ev(5, 110, "completed", {}),
        ],
      },
    },
    { physicalCores: 172 },
  );
  assert.equal(run.footprint.peakCores, 172);
  assert.equal(Math.round(run.footprint.coreSeconds), 344); // 172 cores for 2 s
});
