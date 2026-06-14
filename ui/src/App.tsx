import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileText,
  Play,
  RefreshCw,
  Search,
  Square,
  TerminalSquare,
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8765";

type Health = {
  akaut: {
    baseUrl: string;
    reachable: boolean;
    environmentLoaded: boolean;
    functionCountSample: number;
    error: string | null;
  };
  llm: { model: string; baseUrl: string; hasApiKey: boolean };
  variants: string[];
};

type NodeInfo = {
  name: string;
  qualifiedName: string;
  absolutePath: string;
  type: string;
  line: number;
};

type ConditionItem = {
  nodeId: number | null;
  condition: string;
  lineInFunction: number | null;
  startOffset: number | null;
  endOffset: number | null;
};

type FunctionDetail = {
  absolutePath: string;
  source: string;
  sourceValue: string | null;
  context: string;
  conditions: {
    totalConditions: number;
    totalMcdcPairs: number;
    items: ConditionItem[];
  };
};

type CoverageMetrics = {
  statementCoveragePct?: number;
  branchCoveragePct?: number;
  mcdcCoveragePct?: number;
  coveredMcdcPairs?: number;
  totalMcdcPairs?: number;
  coveredStatements?: number;
  totalStatements?: number;
  coveredBranches?: number;
  totalBranches?: number;
  totalInputTokens?: number;
  totalOutputTokens?: number;
  totalTokens?: number;
  elapsedSec?: number;
  iterationsUsed?: number;
  redundancyRate?: number;
};

type CoverageDetail = {
  visited?: number;
  total?: number;
  progress?: number;
};

type ConditionTrace = {
  nodeId?: number | null;
  condition?: string;
  trueBranchVisited?: boolean;
  falseBranchVisited?: boolean;
  lineInFunction?: number | null;
};

type UnvisitedMcdc = ConditionTrace;

type UnvisitedStatement = {
  nodeId?: number | null;
  statement?: string;
  lineInFunction?: number | null;
};

type UnvisitedBranch = {
  nodeId?: number | null;
  condition?: string;
  trueVisited?: boolean;
  falseVisited?: boolean;
  lineInFunction?: number | null;
};

type RunTest = {
  iteration: number;
  testName: string;
  status: string;
  newMcdcPairsCovered: number;
  isRedundant: boolean;
  testBody?: string;
  executeLog?: string | null;
  elapsedMs?: number;
  tokenInput?: number;
  tokenOutput?: number;
  statementCoverage?: CoverageDetail;
  branchCoverage?: CoverageDetail;
  mcdcCoverage?: CoverageDetail;
  conditionTrace?: ConditionTrace[];
  unvisitedMcdc?: UnvisitedMcdc[];
  unvisitedStatements?: UnvisitedStatement[];
  unvisitedBranches?: UnvisitedBranch[];
  traceSummary?: Record<string, unknown>;
};

type RunState = {
  runId: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  functionPath: string;
  variant: string;
  metrics: CoverageMetrics;
  tests: RunTest[];
  error: string | null;
  resultPath: string | null;
  stopReason?: string | null;
};

type UiEvent = {
  index: number;
  type: string;
  runId: string;
  timestamp: string;
  payload: Record<string, unknown>;
};

type SourceLineReport = {
  lineNumber: number;
  displayLineNumber: number;
  text: string;
  statementStatus: "covered" | "uncovered" | "unknown";
  conditionIds: number[];
  branchIds: number[];
};

type ConditionCoverageReport = {
  nodeId: number;
  condition: string;
  lineInFunction?: number | null;
  startOffset?: number | null;
  endOffset?: number | null;
  trueCovered: boolean;
  falseCovered: boolean;
  missing: string[];
  tests: { iteration: number; testName: string; polarity: string }[];
};

type StatementCoverageReport = {
  nodeId: number;
  statement: string;
  lineInFunction?: number | null;
  covered: boolean;
};

type BranchCoverageReport = {
  nodeId: number;
  condition: string;
  lineInFunction?: number | null;
  startOffset?: number | null;
  endOffset?: number | null;
  trueCovered: boolean;
  falseCovered: boolean;
  missing: string[];
};

type LlmInteraction = {
  callIndex?: number;
  model?: string;
  elapsedMs?: number | null;
  usage?: { promptTokens?: number; completionTokens?: number; totalTokens?: number };
  answer?: string;
  toolCalls?: Array<{ function?: { name?: string; arguments?: string } }>;
};

type RunReport = {
  run: RunState;
  summary: {
    statusCounts: Record<string, number>;
    elapsedSec: number;
    totalInputTokens: number;
    totalOutputTokens: number;
    totalTokens: number;
    stopReason: string | null;
    legacy: boolean;
  };
  source: { lines: SourceLineReport[] };
  conditions: ConditionCoverageReport[];
  statements: StatementCoverageReport[];
  branches: BranchCoverageReport[];
  tests: RunTest[];
  llmInteractions: LlmInteraction[];
};

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [query, setQuery] = useState("XMLElement::FindAttribute");
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [selected, setSelected] = useState<NodeInfo | null>(null);
  const [detail, setDetail] = useState<FunctionDetail | null>(null);
  const [detailTab, setDetailTab] = useState<
    "source" | "conditions" | "context" | "sourceReport" | "testcases" | "summary" | "trace"
  >("source");
  const [runs, setRuns] = useState<RunState[]>([]);
  const [activeRun, setActiveRun] = useState<RunState | null>(null);
  const [report, setReport] = useState<RunReport | null>(null);
  const [selectedTestKey, setSelectedTestKey] = useState<string | null>(null);
  const [events, setEvents] = useState<UiEvent[]>([]);
  const [variant, setVariant] = useState("full");
  const [maxIterations, setMaxIterations] = useState(15);
  const [maxTests, setMaxTests] = useState(15);
  const [mcdcTarget, setMcdcTarget] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    refreshHealth();
    refreshRuns();
    return () => eventSourceRef.current?.close();
  }, []);

  const variants = useMemo(() => health?.variants ?? ["full"], [health]);
  const liveRun = useMemo(
    () => runs.find((run) => isLiveStatus(run.status)) ?? null,
    [runs],
  );
  const visibleRun = activeRun ?? liveRun;

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail ?? `${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  async function refreshHealth() {
    try {
      setHealth(await request<Health>("/api/health"));
    } catch (exc) {
      setError(String(exc));
    }
  }

  async function refreshRuns() {
    try {
      const latestRuns = await request<RunState[]>("/api/runs");
      setRuns(latestRuns);
      const running = latestRuns.find((run) => isLiveStatus(run.status));
      if (running && activeRun?.runId !== running.runId) {
        setActiveRun(running);
        await loadReport(running.runId, false);
        connectEvents(running.runId);
      }
    } catch (exc) {
      setError(String(exc));
    }
  }

  async function searchFunctions(event?: FormEvent) {
    event?.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await request<NodeInfo[]>("/api/akaut/search", {
        method: "POST",
        body: JSON.stringify({ query, types: ["FUNCTION"] }),
      });
      setNodes(result);
      if (result[0]) {
        await selectFunction(result[0]);
      }
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function selectFunction(node: NodeInfo) {
    setSelected(node);
    setDetail(null);
    setError(null);
    try {
      const params = new URLSearchParams({ absolutePath: node.absolutePath });
      setDetail(await request<FunctionDetail>(`/api/akaut/function?${params}`));
      setDetailTab("source");
    } catch (exc) {
      setError(String(exc));
    }
  }

  async function startRun() {
    if (!selected) return;
    if (liveRun || (activeRun && isLiveStatus(activeRun.status))) {
      setError("A run is already active. Cancel or wait for it before starting another run.");
      return;
    }
    setBusy(true);
    setError(null);
    setEvents([]);
    try {
      const run = await request<RunState>("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          absolutePath: selected.absolutePath,
          variant,
          maxIterations,
          maxTests,
          mcdcTarget,
          outDir: "results",
        }),
      });
      setActiveRun(run);
      setReport(null);
      setSelectedTestKey(null);
      await refreshRuns();
      connectEvents(run.runId);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function cancelRun() {
    const runToCancel = activeRun ?? liveRun;
    if (!runToCancel) return;
    try {
      const run = await request<RunState>(`/api/runs/${runToCancel.runId}/cancel`, {
        method: "POST",
      });
      setActiveRun(run);
      await refreshRuns();
    } catch (exc) {
      setError(String(exc));
    }
  }

  async function loadRun(runId: string) {
    try {
      const run = await request<RunState>(`/api/runs/${runId}`);
      setActiveRun(run);
      setEvents([]);
      await loadReport(runId);
    } catch (exc) {
      setError(String(exc));
    }
  }

  async function loadReport(runId: string, surfaceError = true) {
    try {
      const nextReport = await request<RunReport>(`/api/runs/${runId}/report`);
      setReport(nextReport);
      const firstTest = nextReport.tests[0];
      setSelectedTestKey(firstTest ? testKey(firstTest) : null);
      if (nextReport.source.lines.length || nextReport.tests.length) {
        setDetailTab(nextReport.tests.length ? "testcases" : "sourceReport");
      }
    } catch (exc) {
      setReport(null);
      if (surfaceError) setError(String(exc));
    }
  }

  function connectEvents(runId: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`${API_BASE}/api/runs/${runId}/events`);
    eventSourceRef.current = source;
    source.onmessage = (message) => handleEventMessage(message.data);
    [
      "run_started",
      "static_prefetch_completed",
      "test_started",
      "test_completed",
      "coverage_updated",
      "run_completed",
      "run_failed",
      "result_written",
      "log",
    ].forEach((eventName) => {
      source.addEventListener(eventName, (message) =>
        handleEventMessage((message as MessageEvent).data),
      );
    });
    source.onerror = async () => {
      source.close();
      const run = await request<RunState>(`/api/runs/${runId}`).catch(() => null);
      if (run) {
        setActiveRun(run);
        await refreshRuns();
      }
    };
  }

  function handleEventMessage(raw: string) {
    const event = JSON.parse(raw) as UiEvent;
    setEvents((current) => {
      if (current.some((item) => item.index === event.index)) return current;
      return [...current, event].slice(-200);
    });
    const payload = event.payload as Record<string, any>;
    if (event.type === "run_started") {
      setActiveRun((current) =>
        current && current.runId === event.runId
          ? { ...current, status: "running" }
          : current,
      );
    }
    if (event.type === "static_prefetch_completed") {
      setActiveRun((current) =>
        current && current.runId === event.runId
          ? {
              ...current,
              status: "running",
              metrics: {
                ...current.metrics,
                totalMcdcPairs: Number(payload.totalMcdcPairs ?? current.metrics.totalMcdcPairs ?? 0),
                coveredMcdcPairs: current.metrics.coveredMcdcPairs ?? 0,
                statementCoveragePct: current.metrics.statementCoveragePct ?? 0,
                branchCoveragePct: current.metrics.branchCoveragePct ?? 0,
                mcdcCoveragePct: current.metrics.mcdcCoveragePct ?? 0,
              },
            }
          : current,
      );
    }
    if (event.type === "test_completed") {
      setActiveRun((current) => {
        if (!current) return current;
        const test = payload as RunTest & { suite?: CoverageMetrics };
        return {
          ...current,
          status: "running",
          metrics: test.suite ?? current.metrics,
          tests: [...current.tests, test],
        };
      });
      loadReport(event.runId, false);
    }
    if (event.type === "coverage_updated") {
      setActiveRun((current) =>
        current ? { ...current, status: "running", metrics: payload as CoverageMetrics } : current,
      );
      loadReport(event.runId, false);
    }
    if (event.type === "run_completed" || event.type === "run_failed") {
      request<RunState>(`/api/runs/${event.runId}`)
        .then((run) => {
          setActiveRun(run);
          refreshRuns();
          loadReport(event.runId, false);
        })
        .catch(() => undefined);
    }
  }

  function selectTestForNode(nodeId: number) {
    const condition = report?.conditions.find((item) => item.nodeId === nodeId);
    const ref = condition?.tests[0];
    if (!ref) return;
    setSelectedTestKey(`${ref.iteration}-${ref.testName}`);
    setDetailTab("testcases");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <header className="brand">
          <Activity size={22} />
          <div>
            <h1>Covxplore</h1>
            <span>MC/DC generation console</span>
          </div>
        </header>

        <section className="panel compact">
          <div className="panel-title">
            <span>AkaUT</span>
            <button className="icon-button" onClick={refreshHealth} title="Refresh health">
              <RefreshCw size={16} />
            </button>
          </div>
          <StatusPill
            ok={!!health?.akaut.reachable && !!health?.akaut.environmentLoaded}
            text={
              health?.akaut.reachable
                ? health.akaut.environmentLoaded
                  ? "Environment loaded"
                  : "No functions found"
                : "Offline"
            }
          />
          <div className="small-meta">{health?.akaut.baseUrl ?? "http://localhost:8080"}</div>
          <div className="small-meta">{health?.llm.model ?? "model unknown"}</div>
        </section>

        <form className="search-form" onSubmit={searchFunctions}>
          <div className="input-row">
            <Search size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search functions"
            />
          </div>
          <button disabled={busy} type="submit">
            <Search size={16} />
            Search
          </button>
        </form>

        <div className="node-list">
          {nodes.map((node) => (
            <button
              key={`${node.absolutePath}-${node.line}`}
              className={selected?.absolutePath === node.absolutePath ? "node active" : "node"}
              onClick={() => selectFunction(node)}
            >
              <span>{node.qualifiedName}</span>
              <small>line {node.line}</small>
            </button>
          ))}
        </div>
      </aside>

      <main className="workspace">
        {error && (
          <div className="alert">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        <section className="function-header">
          <div>
            <h2>{selected?.qualifiedName ?? "Select a function"}</h2>
            <p>{selected?.absolutePath ?? "AkaUT search results will appear on the left."}</p>
          </div>
          {detail && (
            <div className="metric-strip">
              <Metric label="conditions" value={detail.conditions.totalConditions} />
              <Metric label="MC/DC pairs" value={detail.conditions.totalMcdcPairs} />
              <Metric label="line" value={selected?.line ?? 0} />
            </div>
          )}
        </section>

        <section className="detail-tabs">
          <div className="tabs">
            <button className={detailTab === "source" ? "selected" : ""} onClick={() => setDetailTab("source")}>
              <FileText size={15} /> Source
            </button>
            <button className={detailTab === "conditions" ? "selected" : ""} onClick={() => setDetailTab("conditions")}>
              <CheckCircle2 size={15} /> Conditions
            </button>
            <button className={detailTab === "context" ? "selected" : ""} onClick={() => setDetailTab("context")}>
              <TerminalSquare size={15} /> Context
            </button>
            <button className={detailTab === "sourceReport" ? "selected" : ""} onClick={() => setDetailTab("sourceReport")}>
              <FileText size={15} /> Source Report
            </button>
            <button className={detailTab === "testcases" ? "selected" : ""} onClick={() => setDetailTab("testcases")}>
              <CheckCircle2 size={15} /> Testcases
            </button>
            <button className={detailTab === "summary" ? "selected" : ""} onClick={() => setDetailTab("summary")}>
              <Activity size={15} /> Suite Summary
            </button>
            <button className={detailTab === "trace" ? "selected" : ""} onClick={() => setDetailTab("trace")}>
              <TerminalSquare size={15} /> LLM / Tool Trace
            </button>
          </div>
          <div className="tab-body">
            {!detail && !report && <div className="empty-state">No function or report selected.</div>}
            {detail && detailTab === "source" && <pre>{detail.source}</pre>}
            {detail && detailTab === "context" && <pre>{detail.context}</pre>}
            {detail && detailTab === "conditions" && (
              <div className="condition-table">
                {detail.conditions.items.map((condition) => (
                  <div className="condition-row" key={`${condition.nodeId}-${condition.startOffset}`}>
                    <strong>node:{condition.nodeId ?? "?"}</strong>
                    <span>{condition.condition}</span>
                    <small>line+{condition.lineInFunction ?? "?"}</small>
                  </div>
                ))}
              </div>
            )}
            {detailTab === "sourceReport" && (
              <SourceReportView
                report={report}
                fallbackSource={detail?.source}
                fallbackConditions={detail?.conditions.items ?? []}
                onSelectNode={(nodeId) => selectTestForNode(nodeId)}
              />
            )}
            {detailTab === "testcases" && (
              <TestcaseInspector
                report={report}
                selectedTestKey={selectedTestKey}
                onSelectTest={setSelectedTestKey}
              />
            )}
            {detailTab === "summary" && <SuiteSummary report={report} />}
            {detailTab === "trace" && <TraceView report={report} />}
          </div>
        </section>
      </main>

      <aside className="run-panel">
        <section className="panel">
          <div className="panel-title">
            <span>Run Config</span>
          </div>
          <label>
            Variant
            <select value={variant} onChange={(event) => setVariant(event.target.value)}>
              {variants.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
          <div className="two-col">
            <label>
              Max iter
              <input type="number" min={1} value={maxIterations} onChange={(e) => setMaxIterations(Number(e.target.value))} />
            </label>
            <label>
              Max tests
              <input type="number" min={1} value={maxTests} onChange={(e) => setMaxTests(Number(e.target.value))} />
            </label>
          </div>
          <label>
            MC/DC target
            <input type="number" min={0} max={1} step={0.05} value={mcdcTarget} onChange={(e) => setMcdcTarget(Number(e.target.value))} />
          </label>
          <div className="button-row">
            <button disabled={!selected || busy || Boolean(liveRun || (activeRun && isLiveStatus(activeRun.status)))} onClick={startRun}>
              <Play size={16} />
              Start
            </button>
            <button disabled={!visibleRun || !isLiveStatus(visibleRun.status)} onClick={cancelRun}>
              <Square size={16} />
              Cancel
            </button>
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">
            <span>Live Run</span>
            <StatusPill ok={visibleRun?.status === "completed"} text={visibleRun?.status ?? "idle"} />
          </div>
          <Coverage label="Statement" value={visibleRun?.metrics.statementCoveragePct} />
          <Coverage label="Branch" value={visibleRun?.metrics.branchCoveragePct} />
          <Coverage label="MC/DC" value={visibleRun?.metrics.mcdcCoveragePct} />
          {visibleRun && isLiveStatus(visibleRun.status) && visibleRun.tests.length === 0 && !visibleRun.error && (
            <div className="run-pending">
              <strong>Waiting for first test</strong>
              <span>Static analysis is done; the agent is waiting on the first LLM response before it can call AkaUT.</span>
            </div>
          )}
          {visibleRun?.error && (
            <div className="run-error">
              <strong>Run error</strong>
              <span>{visibleRun.error}</span>
            </div>
          )}
          <div className="run-meta">
            <span>{visibleRun?.runId ?? "No active run"}</span>
            <span>{formatPairs(visibleRun?.metrics)}</span>
          </div>
          <div className="test-table">
            {(visibleRun?.tests ?? []).map((test) => (
              <div className="test-row" key={`${test.iteration}-${test.testName}`}>
                <span>{test.iteration}</span>
                <strong>{test.testName}</strong>
                <StatusPill ok={test.status === "PASSED"} text={test.status} />
                <small>+{test.newMcdcPairsCovered}</small>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">
            <span>Events</span>
          </div>
          <div className="event-log">
            {events.map((event) => (
              <div key={event.index} className="event-line">
                <span>{event.type}</span>
                {typeof event.payload?.message === "string" && <p>{event.payload.message}</p>}
                <small>{new Date(event.timestamp).toLocaleTimeString()}</small>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">
            <span>Results</span>
            <button className="icon-button" onClick={refreshRuns} title="Refresh results">
              <RefreshCw size={16} />
            </button>
          </div>
          <div className="result-list">
            {runs.map((run) => (
              <button key={run.runId} onClick={() => loadRun(run.runId)}>
                <strong>{run.runId}</strong>
                <small>{run.variant} - {pct(run.metrics.mcdcCoveragePct)} MC/DC</small>
              </button>
            ))}
          </div>
        </section>
      </aside>
    </div>
  );
}

function StatusPill({ ok, text }: { ok: boolean; text: string }) {
  return <span className={ok ? "status ok" : "status"}>{text}</span>;
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function Coverage({ label, value }: { label: string; value?: number }) {
  const percent = Math.max(0, Math.min(100, Math.round((value ?? 0) * 100)));
  return (
    <div className="coverage">
      <div>
        <span>{label}</span>
        <strong>{percent}%</strong>
      </div>
      <div className="bar">
        <i style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function SourceReportView({
  report,
  fallbackSource,
  fallbackConditions,
  onSelectNode,
}: {
  report: RunReport | null;
  fallbackSource?: string;
  fallbackConditions: ConditionItem[];
  onSelectNode: (nodeId: number) => void;
}) {
  const reportLines = report?.source.lines ?? [];
  const fallbackLines = fallbackSource ? fallbackSource.split(/\r?\n/) : [];
  const conditionsByLine = new Map<number, ConditionCoverageReport[]>();
  const branchesByLine = new Map<number, BranchCoverageReport[]>();

  for (const condition of report?.conditions ?? []) {
    if (condition.lineInFunction === null || condition.lineInFunction === undefined) continue;
    const existing = conditionsByLine.get(condition.lineInFunction) ?? [];
    existing.push(condition);
    conditionsByLine.set(condition.lineInFunction, existing);
  }
  for (const branch of report?.branches ?? []) {
    if (branch.lineInFunction === null || branch.lineInFunction === undefined) continue;
    const existing = branchesByLine.get(branch.lineInFunction) ?? [];
    existing.push(branch);
    branchesByLine.set(branch.lineInFunction, existing);
  }

  const lines: SourceLineReport[] = reportLines.length
    ? reportLines
    : fallbackLines.map((text, index) => ({
        lineNumber: index,
        displayLineNumber: index + 1,
        text,
        statementStatus: "unknown",
        conditionIds: fallbackConditions
          .filter((item) => item.lineInFunction === index && item.nodeId !== null)
          .map((item) => item.nodeId as number),
        branchIds: [],
      }));

  if (!lines.length) {
    return <div className="empty-state">No source report available.</div>;
  }

  return (
    <div className="report-stack">
      {report?.summary.legacy && (
        <div className="legacy-badge">Legacy result: trace details unavailable</div>
      )}
      <div className="source-legend">
        <span><i className="legend-executed" /> executed line</span>
        <span><b className="tf true">T</b> true branch visited</span>
        <span><b className="tf false">F</b> false branch visited</span>
        <span><b className="tf missing">T</b> not visited</span>
      </div>
      <div className="source-report">
        {lines.map((line) => (
          <SourceCoverageLine
            key={line.lineNumber}
            line={line}
            conditions={conditionsByLine.get(line.lineNumber) ?? []}
            branches={branchesByLine.get(line.lineNumber) ?? []}
            fallbackConditions={fallbackConditions.filter(
              (item) => item.lineInFunction === line.lineNumber,
            )}
            onSelectNode={onSelectNode}
          />
        ))}
      </div>
    </div>
  );
}

function SourceCoverageLine({
  line,
  conditions,
  branches,
  fallbackConditions,
  onSelectNode,
}: {
  line: SourceLineReport;
  conditions: ConditionCoverageReport[];
  branches: BranchCoverageReport[];
  fallbackConditions: ConditionItem[];
  onSelectNode: (nodeId: number) => void;
}) {
  const hasCoveredCondition = conditions.some((item) => item.trueCovered || item.falseCovered);
  const hasCoveredBranch = branches.some((item) => item.trueCovered || item.falseCovered);
  const executed = line.statementStatus === "covered" || hasCoveredCondition || hasCoveredBranch;
  const rowClass = [
    "source-line",
    line.statementStatus,
    executed ? "executed" : "",
  ].join(" ");

  return (
    <div className={rowClass}>
      <span className="line-no">{line.displayLineNumber}</span>
      <span className="stmt-dot" title={`statement ${line.statementStatus}`} />
      <code>{line.text || " "}</code>
      <span className="line-badges">
        {conditions.map((condition) => (
          <CoverageNodeChip
            key={`c-${condition.nodeId}`}
            kind="C"
            nodeId={condition.nodeId}
            label={condition.condition}
            trueCovered={condition.trueCovered}
            falseCovered={condition.falseCovered}
            onSelectNode={onSelectNode}
          />
        ))}
        {branches.map((branch) => (
          <CoverageNodeChip
            key={`b-${branch.nodeId}`}
            kind="B"
            nodeId={branch.nodeId}
            label={branch.condition}
            trueCovered={branch.trueCovered}
            falseCovered={branch.falseCovered}
            onSelectNode={onSelectNode}
          />
        ))}
        {!conditions.length && !branches.length && fallbackConditions.map((condition) => (
          condition.nodeId === null ? null : (
            <CoverageNodeChip
              key={`fallback-${condition.nodeId}`}
              kind="C"
              nodeId={condition.nodeId}
              label={condition.condition}
              trueCovered={false}
              falseCovered={false}
              onSelectNode={onSelectNode}
            />
          )
        ))}
      </span>
    </div>
  );
}

function CoverageNodeChip({
  kind,
  nodeId,
  label,
  trueCovered,
  falseCovered,
  onSelectNode,
}: {
  kind: "C" | "B";
  nodeId: number;
  label: string;
  trueCovered: boolean;
  falseCovered: boolean;
  onSelectNode: (nodeId: number) => void;
}) {
  return (
    <button className="coverage-chip" title={`${kind}:${nodeId} ${label}`} onClick={() => onSelectNode(nodeId)}>
      <span className="node-chip">{kind}:{nodeId}</span>
      <b className={trueCovered ? "tf true" : "tf missing"}>T</b>
      <b className={falseCovered ? "tf false" : "tf missing"}>F</b>
      <small>{label}</small>
    </button>
  );
}

function TestcaseInspector({
  report,
  selectedTestKey,
  onSelectTest,
}: {
  report: RunReport | null;
  selectedTestKey: string | null;
  onSelectTest: (key: string) => void;
}) {
  if (!report?.tests.length) {
    return <div className="empty-state">No testcase data available.</div>;
  }
  const selected =
    report.tests.find((test) => testKey(test) === selectedTestKey) ?? report.tests[0];

  return (
    <div className="inspector-grid">
      <div className="testcase-list">
        {report.tests.map((test) => (
          <button
            className={testKey(test) === testKey(selected) ? "testcase-card selected" : "testcase-card"}
            key={testKey(test)}
            onClick={() => onSelectTest(testKey(test))}
          >
            <span>#{test.iteration}</span>
            <strong>{test.testName}</strong>
            <StatusPill ok={test.status === "PASSED"} text={test.status} />
            <small>
              +{test.newMcdcPairsCovered} MC/DC - {formatMs(test.elapsedMs)} - {formatTokens(test)}
            </small>
          </button>
        ))}
      </div>
      <div className="testcase-detail">
        <div className="detail-head">
          <div>
            <h3>{selected.testName}</h3>
            <p>
              Iteration {selected.iteration} - {formatMs(selected.elapsedMs)} - {formatTokens(selected)}
            </p>
          </div>
          <StatusPill ok={selected.status === "PASSED"} text={selected.status} />
        </div>
        <div className="metric-strip four">
          <Metric label="stmt" value={coverageText(selected.statementCoverage)} />
          <Metric label="branch" value={coverageText(selected.branchCoverage)} />
          <Metric label="MC/DC" value={coverageText(selected.mcdcCoverage)} />
          <Metric label="redundant" value={selected.isRedundant ? "yes" : "no"} />
        </div>
        <section className="subpanel">
          <h4>Test Body</h4>
          <pre>{selected.testBody || "No test body recorded."}</pre>
        </section>
        <section className="subpanel">
          <h4>Execute Log</h4>
          <pre>{selected.executeLog || "No execute log."}</pre>
        </section>
        <section className="subpanel">
          <h4>Condition Trace</h4>
          <ConditionTraceTable items={selected.conditionTrace ?? []} />
        </section>
        <section className="subpanel">
          <h4>Remaining After This Test</h4>
          <RemainingLists test={selected} />
        </section>
      </div>
    </div>
  );
}

function SuiteSummary({ report }: { report: RunReport | null }) {
  if (!report) {
    return <div className="empty-state">No suite report selected.</div>;
  }
  const metrics = report.run.metrics;
  const remainingConditions = report.conditions.filter((item) => item.missing.length);
  const uncoveredStatements = report.statements.filter((item) => !item.covered);
  const uncoveredBranches = report.branches.filter((item) => item.missing.length);

  return (
    <div className="report-stack">
      {report.summary.legacy && (
        <div className="legacy-badge">Legacy result: trace details unavailable</div>
      )}
      <div className="metric-strip four">
        <Metric label="elapsed" value={`${report.summary.elapsedSec || metrics.elapsedSec || 0}s`} />
        <Metric label="tokens" value={report.summary.totalTokens || metrics.totalTokens || 0} />
        <Metric label="input tok" value={report.summary.totalInputTokens || metrics.totalInputTokens || 0} />
        <Metric label="output tok" value={report.summary.totalOutputTokens || metrics.totalOutputTokens || 0} />
      </div>
      <div className="metric-strip four">
        <Metric label="statements" value={`${metrics.coveredStatements ?? 0}/${metrics.totalStatements ?? 0}`} />
        <Metric label="branches" value={`${metrics.coveredBranches ?? 0}/${metrics.totalBranches ?? 0}`} />
        <Metric label="MC/DC" value={`${metrics.coveredMcdcPairs ?? 0}/${metrics.totalMcdcPairs ?? 0}`} />
        <Metric label="stop" value={report.summary.stopReason ?? "n/a"} />
      </div>
      <section className="subpanel">
        <h4>Status Counts</h4>
        <div className="status-grid">
          {Object.entries(report.summary.statusCounts).map(([status, count]) => (
            <span key={status}>
              <strong>{status}</strong>
              {count}
            </span>
          ))}
        </div>
      </section>
      <section className="subpanel">
        <h4>MC/DC Conditions</h4>
        <CoverageMatrix conditions={report.conditions} />
      </section>
      <section className="subpanel">
        <h4>Remaining Coverage</h4>
        <div className="remaining-grid">
          <RemainingBlock title="conditions" items={remainingConditions.map((item) => `node:${item.nodeId} missing ${item.missing.join(", ")}`)} />
          <RemainingBlock title="statements" items={uncoveredStatements.map((item) => `node:${item.nodeId} line+${item.lineInFunction ?? "?"} ${item.statement}`)} />
          <RemainingBlock title="branches" items={uncoveredBranches.map((item) => `node:${item.nodeId} missing ${item.missing.join(", ")}`)} />
        </div>
      </section>
    </div>
  );
}

function TraceView({ report }: { report: RunReport | null }) {
  if (!report?.llmInteractions.length) {
    return <div className="empty-state">No LLM/tool trace available.</div>;
  }
  return (
    <div className="trace-list">
      {report.llmInteractions.map((item, index) => (
        <details key={`${item.callIndex ?? index}-${item.model ?? "model"}`}>
          <summary>
            <strong>Call {item.callIndex ?? index + 1}</strong>
            <span>{item.model ?? "unknown model"} - {formatMs(item.elapsedMs ?? undefined)} - {item.usage?.totalTokens ?? 0} tokens</span>
          </summary>
          {item.toolCalls?.length ? (
            <div className="tool-call-list">
              {item.toolCalls.map((tool, toolIndex) => (
                <pre key={toolIndex}>
                  {tool.function?.name ?? "tool"} {tool.function?.arguments ?? ""}
                </pre>
              ))}
            </div>
          ) : null}
          <pre>{item.answer || "No assistant answer recorded."}</pre>
        </details>
      ))}
    </div>
  );
}

function ConditionTraceTable({ items }: { items: ConditionTrace[] }) {
  if (!items.length) return <div className="empty-inline">No condition trace recorded.</div>;
  return (
    <div className="trace-report">
      {items.map((item, index) => (
        <ConditionReportRow
          key={`${item.nodeId}-${index}`}
          nodeId={item.nodeId}
          lineInFunction={item.lineInFunction}
          condition={item.condition ?? ""}
          trueVisited={Boolean(item.trueBranchVisited)}
          falseVisited={Boolean(item.falseBranchVisited)}
          mode="visited"
        />
      ))}
    </div>
  );
}

function RemainingLists({ test }: { test: RunTest }) {
  const mcdc = test.unvisitedMcdc ?? [];
  const statements = test.unvisitedStatements ?? [];
  const branches = test.unvisitedBranches ?? [];
  if (!mcdc.length && !statements.length && !branches.length) {
    return <div className="empty-inline">No remaining obligations after this test.</div>;
  }

  return (
    <div className="remaining-report">
      <MiniReportSection title="MC/DC conditions" count={mcdc.length}>
        {mcdc.map((item, index) => (
          <ConditionReportRow
            key={`mcdc-${item.nodeId}-${index}`}
            nodeId={item.nodeId}
            lineInFunction={item.lineInFunction}
            condition={item.condition ?? ""}
            trueVisited={Boolean(item.trueBranchVisited)}
            falseVisited={Boolean(item.falseBranchVisited)}
            mode="remaining"
          />
        ))}
      </MiniReportSection>
      <MiniReportSection title="Statements" count={statements.length}>
        {statements.map((item, index) => (
          <StatementReportRow
            key={`stmt-${item.nodeId}-${index}`}
            nodeId={item.nodeId}
            lineInFunction={item.lineInFunction}
            statement={item.statement ?? ""}
          />
        ))}
      </MiniReportSection>
      <MiniReportSection title="Branches" count={branches.length}>
        {branches.map((item, index) => (
          <ConditionReportRow
            key={`branch-${item.nodeId}-${index}`}
            nodeId={item.nodeId}
            lineInFunction={item.lineInFunction}
            condition={item.condition ?? ""}
            trueVisited={Boolean(item.trueVisited)}
            falseVisited={Boolean(item.falseVisited)}
            mode="remaining"
          />
        ))}
      </MiniReportSection>
    </div>
  );
}

function MiniReportSection({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: ReactNode;
}) {
  return (
    <div className="mini-report-section">
      <div className="mini-report-head">
        <strong>{title}</strong>
        <span>{count}</span>
      </div>
      <div className="trace-report">{count ? children : <div className="empty-inline">none</div>}</div>
    </div>
  );
}

function ConditionReportRow({
  nodeId,
  lineInFunction,
  condition,
  trueVisited,
  falseVisited,
  mode,
}: {
  nodeId?: number | null;
  lineInFunction?: number | null;
  condition: string;
  trueVisited: boolean;
  falseVisited: boolean;
  mode: "visited" | "remaining";
}) {
  return (
    <div className="trace-row">
      <div className="node-cell">
        <strong>node:{nodeId ?? "?"}</strong>
        <small>line+{lineInFunction ?? "?"}</small>
      </div>
      <div className="trace-condition">
        <code>{condition || "condition unavailable"}</code>
      </div>
      <div className="tf-pair" title={mode === "remaining" ? "colored = already seen, gray = still missing" : "colored = visited in this test"}>
        <b className={trueVisited ? "tf true" : "tf missing"}>T</b>
        <b className={falseVisited ? "tf false" : "tf missing"}>F</b>
      </div>
    </div>
  );
}

function StatementReportRow({
  nodeId,
  lineInFunction,
  statement,
}: {
  nodeId?: number | null;
  lineInFunction?: number | null;
  statement: string;
}) {
  return (
    <div className="trace-row statement-missing">
      <div className="node-cell">
        <strong>node:{nodeId ?? "?"}</strong>
        <small>line+{lineInFunction ?? "?"}</small>
      </div>
      <div className="trace-condition">
        <code>{statement || "statement unavailable"}</code>
      </div>
      <div className="statement-badge">uncovered</div>
    </div>
  );
}

function CoverageMatrix({ conditions }: { conditions: ConditionCoverageReport[] }) {
  if (!conditions.length) return <div className="empty-inline">No condition details available.</div>;
  return (
    <div className="coverage-table">
      {conditions.map((item) => (
        <div key={item.nodeId}>
          <strong>node:{item.nodeId}</strong>
          <span>{item.condition}</span>
          <small className="tf-pair">
            <b className={item.trueCovered ? "tf true" : "tf missing"}>T</b>
            <b className={item.falseCovered ? "tf false" : "tf missing"}>F</b>
          </small>
        </div>
      ))}
    </div>
  );
}

function RemainingBlock({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="remaining-block">
      <strong>{title}</strong>
      {items.length ? items.slice(0, 12).map((item, index) => <span key={`${title}-${index}`}>{item}</span>) : <span>none</span>}
      {items.length > 12 && <small>+{items.length - 12} more</small>}
    </div>
  );
}

function pct(value?: number) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function formatPairs(metrics?: CoverageMetrics) {
  if (!metrics?.totalMcdcPairs) return "0/0 MC/DC";
  return `${metrics.coveredMcdcPairs ?? 0}/${metrics.totalMcdcPairs} MC/DC`;
}

function isLiveStatus(status: RunState["status"]) {
  return status === "queued" || status === "running";
}

function testKey(test: Pick<RunTest, "iteration" | "testName">) {
  return `${test.iteration}-${test.testName}`;
}

function coverageText(cov?: CoverageDetail) {
  if (!cov) return "0/0";
  return `${cov.visited ?? 0}/${cov.total ?? 0}`;
}

function formatMs(value?: number | null) {
  if (!value) return "0ms";
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}

function formatTokens(test: Pick<RunTest, "tokenInput" | "tokenOutput">) {
  return `${(test.tokenInput ?? 0) + (test.tokenOutput ?? 0)} tok`;
}
