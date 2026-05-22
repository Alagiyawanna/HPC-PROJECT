import React, { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";
const METHOD_ORDER = ["serial", "openmp", "mpi", "cuda"];

const METHOD_META = {
  serial: {
    label: "Serial",
    buttonLabel: "Run Serial",
    accent: "from-rose-500 via-orange-500 to-amber-400",
    border: "border-rose-500/30",
    hint: "Single-core baseline",
  },
  openmp: {
    label: "OpenMP",
    buttonLabel: "Run OpenMP",
    accent: "from-cyan-500 via-sky-500 to-blue-500",
    border: "border-cyan-500/30",
    hint: "Shared-memory parallelism",
  },
  mpi: {
    label: "MPI",
    buttonLabel: "Run MPI",
    accent: "from-violet-500 via-fuchsia-500 to-pink-500",
    border: "border-violet-500/30",
    hint: "Distributed-memory parallelism",
  },
  cuda: {
    label: "CUDA",
    buttonLabel: "Run CUDA",
    accent: "from-emerald-500 via-teal-500 to-cyan-500",
    border: "border-emerald-500/30",
    hint: "GPU acceleration",
  },
};

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Pending";
  }

  if (value < 1) {
    return `${(value * 1000).toFixed(2)} ms`;
  }

  return `${value.toFixed(6)} s`;
}

function ResultStat({ label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 shadow-sm dark:border-white/10 dark:bg-white/5">
      <p className="text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-slate-400">
        {label}
      </p>
      <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
        {value}
      </p>
    </div>
  );
}

function OutputCard({ methodKey, loadingMethod, result, onRun, serialTime }) {
  const meta = METHOD_META[methodKey];
  const isLoading = loadingMethod === methodKey;
  const speedup =
    methodKey === "serial"
      ? 1
      : serialTime && result?.executionTimeSeconds
        ? serialTime / result.executionTimeSeconds
        : null;

  return (
    <article
      className={`glass-panel rounded-[2rem] border ${meta.border} bg-white/70 p-4 shadow-glow transition dark:bg-slate-950/70`}
    >
      <div className="flex items-center justify-between gap-3 pb-4">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">
            {meta.label}
          </p>
          <h3 className="mt-1 text-xl font-bold text-slate-900 dark:text-white">
            Output Image
          </h3>
        </div>
        <button
          type="button"
          onClick={() => onRun(methodKey)}
          disabled={isLoading}
          className={`inline-flex items-center gap-2 rounded-full bg-gradient-to-r ${meta.accent} px-4 py-2 text-sm font-semibold text-white shadow-lg transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-70`}
        >
          {isLoading ? "Running..." : meta.buttonLabel}
        </button>
      </div>

      <div className="overflow-hidden rounded-[1.5rem] border border-white/10 bg-slate-950/80">
        {result?.imageUrl ? (
          <img
            src={result.imageUrl}
            alt={`${meta.label} output`}
            className="aspect-[4/3] w-full object-cover"
          />
        ) : (
          <div className="flex aspect-[4/3] items-center justify-center px-6 text-center text-slate-400">
            <div>
              <p className="text-base font-medium text-slate-200">
                No result yet
              </p>
              <p className="mt-2 text-sm">
                Click {meta.buttonLabel} to run the executable and load this
                panel.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        <ResultStat
          label="Execution Time"
          value={formatSeconds(result?.executionTimeSeconds)}
        />
        <ResultStat
          label="Speedup vs Serial"
          value={
            methodKey === "serial"
              ? "1.00x baseline"
              : serialTime && result?.executionTimeSeconds
                ? `${speedup.toFixed(2)}x`
                : "Run Serial first"
          }
        />
      </div>

      <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">
        {meta.hint}
      </p>
    </article>
  );
}

function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("theme") || "dark",
  );
  const [loadingMethod, setLoadingMethod] = useState(null);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState(
    "Run each implementation to compare execution time and output quality.",
  );
  const [results, setResults] = useState({
    serial: null,
    openmp: null,
    mpi: null,
    cuda: null,
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.classList.toggle("light", theme === "light");
    localStorage.setItem("theme", theme);
  }, [theme]);

  const serialTime = results.serial?.executionTimeSeconds ?? null;

  const chartData = useMemo(
    () =>
      METHOD_ORDER.map((methodKey) => ({
        name: METHOD_META[methodKey].label,
        time: results[methodKey]?.executionTimeSeconds ?? 0,
        hasValue: Boolean(results[methodKey]),
      })),
    [results],
  );

  async function runMethod(methodKey) {
    setLoadingMethod(methodKey);
    setError("");
    setStatusMessage(`Running ${METHOD_META[methodKey].label}...`);

    try {
      const response = await fetch(`${API_BASE_URL}/api/run/${methodKey}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const payload = await response.json();

      if (!response.ok || !payload.success) {
        throw new Error(
          payload.error || `Failed to run ${METHOD_META[methodKey].label}`,
        );
      }

      setResults((current) => ({
        ...current,
        [methodKey]: {
          executionTimeSeconds: Number(payload.executionTimeSeconds),
          executionTimeLabel: payload.executionTimeLabel,
          imageUrl: `${API_BASE_URL}${payload.imageUrl}?v=${Date.now()}`,
          stdout: payload.stdout,
        },
      }));

      setStatusMessage(
        payload.message ||
          `${METHOD_META[methodKey].label} completed successfully.`,
      );
    } catch (requestError) {
      setError(requestError.message || "An unexpected error occurred.");
      setStatusMessage("Execution failed.");
    } finally {
      setLoadingMethod(null);
    }
  }

  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-6 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <section className="glass-panel relative overflow-hidden rounded-[2.5rem] border border-white/10 bg-slate-950/70 px-6 py-6 shadow-glow sm:px-8 lg:px-10">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_rgba(56,189,248,0.18),_transparent_32%),radial-gradient(circle_at_bottom_left,_rgba(168,85,247,0.14),_transparent_26%)]" />
          <div className="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-1 text-xs font-semibold uppercase tracking-[0.34em] text-slate-300">
                HPC Image Convolution Demo
              </p>
              <h1 className="mt-4 text-4xl font-black tracking-tight text-white sm:text-5xl lg:text-6xl">
                Compare Serial, OpenMP, MPI, and CUDA in one clean dashboard.
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
                Execute each compiled binary from the browser, inspect its
                output image, and track performance updates in real time with a
                live bar chart.
              </p>
            </div>

            <button
              type="button"
              onClick={() =>
                setTheme((currentTheme) =>
                  currentTheme === "dark" ? "light" : "dark",
                )
              }
              className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/10 px-5 py-3 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15"
            >
              {theme === "dark"
                ? "Switch to Light Mode"
                : "Switch to Dark Mode"}
            </button>
          </div>
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <article className="glass-panel rounded-[2rem] border border-white/10 bg-white/75 p-5 shadow-glow dark:bg-slate-950/70">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">
                  Input
                </p>
                <h2 className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">
                  Base Input Image
                </h2>
              </div>
              <div className="rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm font-medium text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                Static reference for all runs
              </div>
            </div>

            <div className="mt-5 overflow-hidden rounded-[1.5rem] border border-white/10 bg-slate-950">
              <img
                src={`${API_BASE_URL}/image.png`}
                alt="Input image"
                className="aspect-[4/3] w-full object-cover"
              />
            </div>
          </article>

          <article className="glass-panel rounded-[2rem] border border-white/10 bg-white/75 p-5 shadow-glow dark:bg-slate-950/70">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">
                  Control Panel
                </p>
                <h2 className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">
                  Run Comparisons
                </h2>
              </div>
              <div className="rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm font-medium text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                {statusMessage}
              </div>
            </div>

            {error ? (
              <div className="mt-5 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {error}
              </div>
            ) : null}

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              {METHOD_ORDER.map((methodKey) => {
                const meta = METHOD_META[methodKey];
                const isLoading = loadingMethod === methodKey;

                return (
                  <button
                    key={methodKey}
                    type="button"
                    onClick={() => runMethod(methodKey)}
                    disabled={Boolean(loadingMethod)}
                    className={`group relative overflow-hidden rounded-[1.5rem] border ${meta.border} bg-white/10 px-5 py-5 text-left transition hover:-translate-y-0.5 hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-white/5 dark:hover:bg-white/10`}
                  >
                    <div
                      className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${meta.accent}`}
                    />
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-xs uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">
                          {meta.label}
                        </p>
                        <h3 className="mt-2 text-lg font-bold text-slate-900 dark:text-white">
                          {meta.buttonLabel}
                        </h3>
                        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
                          {meta.hint}
                        </p>
                      </div>
                      <span className="rounded-full border border-white/10 bg-slate-950/70 px-3 py-1 text-xs font-semibold text-slate-100">
                        {isLoading ? "Running" : "Ready"}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </article>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-2">
          {METHOD_ORDER.map((methodKey) => (
            <OutputCard
              key={methodKey}
              methodKey={methodKey}
              loadingMethod={loadingMethod}
              result={results[methodKey]}
              onRun={runMethod}
              serialTime={serialTime}
            />
          ))}
        </section>

        <section className="glass-panel mt-6 rounded-[2rem] border border-white/10 bg-white/75 p-5 shadow-glow dark:bg-slate-950/70">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">
                Analysis
              </p>
              <h2 className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">
                Execution Time Comparison
              </h2>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Bars update automatically when a method finishes. Missing values
              are shown as zero until executed.
            </p>
          </div>

          <div className="mt-6 h-[360px] w-full rounded-[1.5rem] border border-white/10 bg-slate-950/80 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 10, right: 20, bottom: 10, left: 0 }}
              >
                <defs>
                  <linearGradient
                    id="chartGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopColor="#38bdf8" stopOpacity={1} />
                    <stop
                      offset="100%"
                      stopColor="#8b5cf6"
                      stopOpacity={0.72}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148,163,184,0.22)"
                />
                <XAxis
                  dataKey="name"
                  stroke="#94a3b8"
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `${Number(value).toFixed(2)}`}
                  label={{
                    value: "Seconds",
                    angle: -90,
                    position: "insideLeft",
                    fill: "#94a3b8",
                  }}
                />
                <Tooltip
                  cursor={{ fill: "rgba(148,163,184,0.08)" }}
                  contentStyle={{
                    background: "rgba(15, 23, 42, 0.96)",
                    border: "1px solid rgba(148,163,184,0.22)",
                    borderRadius: "16px",
                    color: "#e2e8f0",
                  }}
                  formatter={(value, name, entry) => [
                    entry.payload.hasValue
                      ? `${Number(value).toFixed(6)} s`
                      : "Pending",
                    "Execution Time",
                  ]}
                />
                <Bar
                  dataKey="time"
                  radius={[12, 12, 0, 0]}
                  fill="url(#chartGradient)"
                >
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={
                        entry.hasValue
                          ? "url(#chartGradient)"
                          : "rgba(148, 163, 184, 0.24)"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </main>
  );
}

export default App;
