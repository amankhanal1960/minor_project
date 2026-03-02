type Severity = "Low" | "Medium" | "High";

type CoughEvent = {
  id: number;
  timestamp: string;
  confidence: number;
  motionDetected: boolean;
  durationMs: number;
  severity: Severity;
};

const coughEvents: CoughEvent[] = [
  {
    id: 1,
    timestamp: "2026-03-02 14:27:18",
    confidence: 0.93,
    motionDetected: true,
    durationMs: 670,
    severity: "High",
  },
  {
    id: 2,
    timestamp: "2026-03-02 13:11:52",
    confidence: 0.87,
    motionDetected: true,
    durationMs: 540,
    severity: "Medium",
  },
  {
    id: 3,
    timestamp: "2026-03-02 11:44:09",
    confidence: 0.78,
    motionDetected: true,
    durationMs: 460,
    severity: "Low",
  },
  {
    id: 4,
    timestamp: "2026-03-02 09:58:36",
    confidence: 0.90,
    motionDetected: true,
    durationMs: 620,
    severity: "High",
  },
  {
    id: 5,
    timestamp: "2026-03-02 08:36:04",
    confidence: 0.82,
    motionDetected: true,
    durationMs: 500,
    severity: "Medium",
  },
];

const hourlyCount = [
  { hour: "08", count: 1 },
  { hour: "09", count: 1 },
  { hour: "10", count: 0 },
  { hour: "11", count: 1 },
  { hour: "12", count: 0 },
  { hour: "13", count: 1 },
  { hour: "14", count: 1 },
];

const totalCoughs = coughEvents.length;
const averageConfidence = Math.round(
  (coughEvents.reduce((sum, event) => sum + event.confidence, 0) / totalCoughs) * 100,
);
const motionConfirmedCount = coughEvents.filter((event) => event.motionDetected).length;
const latestTimestamp = coughEvents[0]?.timestamp ?? "No event";
const maxHourly = Math.max(...hourlyCount.map((entry) => entry.count), 1);

const severityClassMap: Record<Severity, string> = {
  High: "event-chip-high",
  Medium: "event-chip-med",
  Low: "event-chip-low",
};

export default function Home() {
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
                Static UI mock for real-time monitoring. Counts and timestamps are shown from
                sample data now, then this same shape can be fed by WebSocket events.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <span className="live-badge">
                <span className="live-dot" />
                Static Feed (WebSocket-ready)
              </span>
              <div className="rounded-full border border-[color:var(--line)] bg-white px-4 py-2 text-sm">
                Last update: <span className="font-mono">{latestTimestamp}</span>
              </div>
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <article className="surface reveal-1 p-5">
            <p className="section-title">Total Coughs</p>
            <p className="value mt-3 text-4xl">{totalCoughs}</p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">Current day</p>
          </article>

          <article className="surface reveal-1 p-5">
            <p className="section-title">Avg Audio Confidence</p>
            <p className="value mt-3 text-4xl">{averageConfidence}%</p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">From accepted events</p>
          </article>

          <article className="surface reveal-2 p-5">
            <p className="section-title">Motion Confirmed</p>
            <p className="value mt-3 text-4xl">{motionConfirmedCount}</p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">Decision-level fusion pass</p>
          </article>

          <article className="surface reveal-2 p-5">
            <p className="section-title">System State</p>
            <p className="value mt-3 text-3xl text-[color:var(--good)]">Monitoring</p>
            <p className="mt-2 text-sm text-[color:var(--muted)]">No stream errors (mock)</p>
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
              {coughEvents.map((event) => (
                <div
                  key={event.id}
                  className="event-item grid gap-3 p-4 md:grid-cols-[1.5fr_1fr_1fr_0.8fr] md:items-center"
                >
                  <div>
                    <p className="text-xs text-[color:var(--muted)]">Timestamp</p>
                    <p className="font-mono text-sm font-semibold sm:text-base">{event.timestamp}</p>
                  </div>

                  <div>
                    <p className="text-xs text-[color:var(--muted)]">Confidence</p>
                    <p className="font-mono text-sm font-semibold">{Math.round(event.confidence * 100)}%</p>
                  </div>

                  <div>
                    <p className="text-xs text-[color:var(--muted)]">Duration</p>
                    <p className="font-mono text-sm font-semibold">{event.durationMs} ms</p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 md:justify-end">
                    <span className={`event-chip ${severityClassMap[event.severity]}`}>
                      {event.severity}
                    </span>
                    <span className="event-chip bg-[#e9f4ef] text-[#205a43]">Motion: yes</span>
                  </div>
                </div>
              ))}
            </div>
          </article>

          <aside className="space-y-5">
            <article className="surface reveal-3 p-5 sm:p-6">
              <h2 className="mb-4 text-xl font-semibold">Hourly Trend</h2>
              <div className="grid-bars">
                {hourlyCount.map((entry) => (
                  <div key={entry.hour} className="bar-row">
                    <span className="font-mono text-xs text-[color:var(--muted)]">{entry.hour}:00</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${(entry.count / maxHourly) * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs font-semibold">{entry.count}</span>
                  </div>
                ))}
              </div>
            </article>

            <article className="surface reveal-3 p-5 sm:p-6">
              <h2 className="text-xl font-semibold">WebSocket Mapping</h2>
              <p className="mt-3 text-sm text-[color:var(--muted)]">
                Keep this event payload shape when switching to live data:
              </p>
              <pre className="mt-4 overflow-x-auto rounded-xl border border-[color:var(--line)] bg-[#f7faf4] p-3 text-xs leading-relaxed text-[#264033]">
{`{
  "timestamp": "2026-03-02 14:27:18",
  "confidence": 0.93,
  "motionDetected": true,
  "durationMs": 670,
  "severity": "High"
}`}
              </pre>
            </article>
          </aside>
        </section>
      </main>
    </div>
  );
}
