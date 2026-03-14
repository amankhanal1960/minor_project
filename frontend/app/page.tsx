const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:5000/api";

type Severity = "Low" | "Medium" | "High";

type Detection = {
  id: string;
  deviceId: string;
  coughProbability: number;
  audioLevel: number;
  detectionAt: string;
};

type EventVM = {
  id: string;
  timestamp: string;
  probabilityPct: number;
  audioLevel: number;
  deviceId: string;
  severity: Severity;
  rawDate: string;
};

function classifySeverity(probability: number): Severity {
  if (probability >= 0.8) return "High";
  if (probability >= 0.5) return "Medium";
  return "Low";
}

function formatTime(value: string) {
  return new Date(value).toLocaleString();
}

async function fetchDetections(): Promise<Detection[]> {
  try {
    const res = await fetch(`${API_BASE}/detections`, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to fetch detections");
    return (await res.json()) as Detection[];
  } catch (err) {
    console.error("Failed to load detections", err);
    return [];
  }
}

function buildViewModel(detections: Detection[]): {
  events: EventVM[];
  hourly: { hour: string; count: number }[];
  summary: {
    total: number;
    avgProbabilityPct: number;
    avgAudio: number;
    latest: string;
  };
} {
  const events = detections
    .sort(
      (a, b) =>
        new Date(b.detectionAt).getTime() - new Date(a.detectionAt).getTime(),
    )
    .map((d) => ({
      id: d.id,
      timestamp: formatTime(d.detectionAt),
      rawDate: d.detectionAt,
      probabilityPct: Math.round(d.coughProbability * 100),
      audioLevel: d.audioLevel,
      deviceId: d.deviceId,
      severity: classifySeverity(d.coughProbability),
    }));

  const total = events.length;
  const avgProbabilityPct =
    total === 0
      ? 0
      : Math.round(
          events.reduce((sum, e) => sum + e.probabilityPct, 0) / total,
        );
  const avgAudio =
    total === 0
      ? 0
      : Math.round(events.reduce((sum, e) => sum + e.audioLevel, 0) / total);
  const latest = events[0]?.timestamp ?? "No data yet";

  const hourlyMap = new Map<string, number>();
  events.forEach((e) => {
    const hour = new Date(e.rawDate).getHours().toString().padStart(2, "0");
    hourlyMap.set(hour, (hourlyMap.get(hour) ?? 0) + 1);
  });
  const hourly = Array.from({ length: 24 }, (_, i) => {
    const hour = i.toString().padStart(2, "0");
    return { hour, count: hourlyMap.get(hour) ?? 0 };
  }).filter((entry) => entry.count > 0);

  return {
    events,
    hourly,
    summary: { total, avgProbabilityPct, avgAudio, latest },
  };
}

const severityClassMap: Record<Severity, string> = {
  High: "event-chip-high",
  Medium: "event-chip-med",
  Low: "event-chip-low",
};

export default async function Home() {
  const detections = await fetchDetections();
  const { events, hourly, summary } = buildViewModel(detections);
  const maxHourly = Math.max(...hourly.map((entry) => entry.count), 1);

  return (
    <div className="min-h-screen px-4 py-6 sm:px-8 sm:py-10">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-5 sm:gap-6">
        <section className="surface reveal-1 p-5 sm:p-7">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-3">
              <p className="section-title">Cough Monitoring Console</p>
              <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
                Patient Cough Activity
              </h1>
              <p className="max-w-2xl text-sm text-[color:var(--muted)] sm:text-base">
                Live data from backend detections. Add a device and POST
                detections to see them here.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <span className="live-badge">
                <span className="live-dot" />
                Live Feed (poll on load)
              </span>
              <div className="rounded-full border border-[color:var(--line)] bg-white px-4 py-2 text-sm">
                Last update: <span className="font-mono">{summary.latest}</span>
              </div>
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <article className="surface reveal-1 p-5">
            <p className="section-title">Total Coughs</p>
            <p className="value mt-3 text-4xl">{summary.total}</p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">
              Current day
            </p>
          </article>

          <article className="surface reveal-1 p-5">
            <p className="section-title">Avg Cough Probability</p>
            <p className="value mt-3 text-4xl">{summary.avgProbabilityPct}%</p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">
              From detections received
            </p>
          </article>

          <article className="surface reveal-2 p-5">
            <p className="section-title">Avg Audio Level</p>
            <p className="value mt-3 text-4xl">{summary.avgAudio}</p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">
              Linear scale from payload
            </p>
          </article>

          <article className="surface reveal-2 p-5">
            <p className="section-title">System State</p>
            <p className="value mt-3 text-3xl text-[color:var(--good)]">
              Monitoring
            </p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">
              Backend reachable
            </p>
          </article>
        </section>

        <section className="grid grid-cols-1 gap-5 lg:grid-cols-[1.25fr_0.75fr]">
          <article className="surface reveal-2 p-5 sm:p-6">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Detected Cough Events</h2>
              <span className="rounded-full bg-[color:var(--accent-soft)] px-3 py-1 text-xs font-semibold text-[color:var(--accent)]">
                Newest first
              </span>
            </div>

            <div className="space-y-3">
              {events.map((event) => (
                <div
                  key={event.id}
                  className="event-item grid gap-3 p-4 md:grid-cols-[1.5fr_1fr_1fr_0.8fr] md:items-center"
                >
                  <div>
                    <p className="text-xs text-[color:var(--muted)]">
                      Timestamp
                    </p>
                    <p className="font-mono text-sm font-semibold sm:text-base">
                      {event.timestamp}
                    </p>
                  </div>

                  <div>
                    <p className="text-xs text-[color:var(--muted)]">
                      Cough Probability
                    </p>
                    <p className="font-mono text-sm font-semibold">
                      {event.probabilityPct}%
                    </p>
                  </div>

                  <div>
                    <p className="text-xs text-[color:var(--muted)]">
                      Audio Level
                    </p>
                    <p className="font-mono text-sm font-semibold">
                      {event.audioLevel}
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 md:justify-end">
                    <span
                      className={`event-chip ${severityClassMap[event.severity]}`}
                    >
                      {event.severity}
                    </span>
                    <span className="event-chip bg-[#e9f4ef] text-[#205a43]">
                      Device: {event.deviceId}
                    </span>
                  </div>
                </div>
              ))}

              {events.length === 0 && (
                <div className="rounded-lg border border-dashed border-[color:var(--line)] p-6 text-sm text-[color:var(--muted)]">
                  No detections yet. POST to <code>/api/detections</code> to see
                  data appear.
                </div>
              )}
            </div>
          </article>

          <aside className="space-y-5">
            <article className="surface reveal-3 p-5 sm:p-6">
              <h2 className="mb-4 text-xl font-semibold">Hourly Trend</h2>
              <div className="grid-bars">
                {hourly.map((entry) => (
                  <div key={entry.hour} className="bar-row">
                    <span className="font-mono text-xs text-[color:var(--muted)]">
                      {entry.hour}:00
                    </span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${(entry.count / maxHourly) * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs font-semibold">
                      {entry.count}
                    </span>
                  </div>
                ))}

                {hourly.length === 0 && (
                  <p className="text-sm text-[color:var(--muted)]">
                    No detections to plot yet.
                  </p>
                )}
              </div>
            </article>

            <article className="surface reveal-3 p-5 sm:p-6">
              <h2 className="text-xl font-semibold">WebSocket Mapping</h2>
              <p className="mt-3 text-sm text-[color:var(--muted)]">
                Keep this event payload shape when switching to live data:
              </p>
              <pre className="mt-4 overflow-x-auto rounded-xl border border-[color:var(--line)] bg-[#f7faf4] p-3 text-xs leading-relaxed text-[#264033]">
                {`{
  "deviceId": "device-1234",
  "coughProbability": 0.82,
  "audioLevel": 89,
  "detectionAt": "2026-03-13T09:56:45.384Z"
}`}
              </pre>
            </article>
          </aside>
        </section>
      </main>
    </div>
  );
}
