import { useEffect, useMemo, useRef, useState } from 'react';

type StartEvent = {
  run: string;
  event: 'run_started';
  steps: number;
  seed: number;
  t_ms: number;
};

type StepEvent = {
  run: string;
  event: 'step';
  step: number;
  loss: number;
  elapsed_ms: number;
  t_ms: number;
};

type TensorEvent = {
  run: string;
  event: 'tensor';
  step: number;
  name: string;
  shape: number[];
  values: number[];
  t_ms: number;
};

type OtherEvent = {
  run: string;
  event: 'checkpoint' | 'run_completed' | 'run_failed';
  path?: string;
  steps?: number;
  loss?: number;
  message?: string;
  t_ms: number;
};

export type TraceEvent = StartEvent | StepEvent | TensorEvent | OtherEvent;

type Props = {
  /** Events parsed at build time, so the page renders without a network call. */
  initialEvents?: TraceEvent[];
  /** Fallback path of the bundled demo trace. */
  source?: string;
};

export default function ResearchStudio({
  initialEvents = [],
  source = '/sample-trace.jsonl',
}: Props) {
  const [events, setEvents] = useState<TraceEvent[]>(initialEvents);
  const [error, setError] = useState<string | null>(null);
  const [label, setLabel] = useState('sample-trace.jsonl');
  const [selected, setSelected] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const hasInitial = initialEvents.length > 0;

  useEffect(() => {
    if (hasInitial) return;
    let cancelled = false;
    fetch(source)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.text();
      })
      .then((text) => {
        if (!cancelled) {
          setEvents(parseTrace(text));
          setLabel(source.split('/').pop() ?? source);
        }
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(`无法加载示例轨迹：${reason.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, [hasInitial, source]);

  const steps = useMemo(
    () => events.filter((event): event is StepEvent => event.event === 'step'),
    [events],
  );
  const tensors = useMemo(
    () => events.filter((event): event is TensorEvent => event.event === 'tensor'),
    [events],
  );
  const attention = useMemo(
    () => tensors.filter((event) => event.name.startsWith('attention.')),
    [tensors],
  );
  const logits = useMemo(
    () => tensors.find((event) => event.name === 'logits') ?? null,
    [tensors],
  );
  const run = events[0]?.run ?? '—';
  const finished = events.find(
    (event): event is OtherEvent => event.event === 'run_completed',
  );
  const selectedTensor =
    attention.find((event) => event.name === selected) ?? attention[0] ?? null;

  useEffect(() => {
    if (!selected && attention.length > 0) setSelected(attention[0].name);
  }, [attention, selected]);

  function loadFile(file: File) {
    file
      .text()
      .then((text) => {
        try {
          setEvents(parseTrace(text));
          setLabel(file.name);
          setError(null);
        } catch (reason) {
          setError(`无法解析 ${file.name}：${(reason as Error).message}`);
        }
      })
      .catch((reason: Error) => setError(`读取失败：${reason.message}`));
  }

  return (
    <div className="studio">
      <header className="studio__bar">
        <div>
          <p className="studio__eyebrow">Research Studio</p>
          <h1>把一次训练拆开看</h1>
          <p className="studio__lead">
            读取 <code>cargo run -p mini-llm-cli -- trace</code> 写出的 JSONL：损失曲线、每个
            head 的注意力矩阵、最后一个位置的 logits。
          </p>
        </div>
        <div className="studio__actions">
          <input
            ref={fileInput}
            type="file"
            accept=".jsonl,.json,application/json,text/plain"
            className="studio__file"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) loadFile(file);
            }}
          />
          <button type="button" className="studio__btn" onClick={() => fileInput.current?.click()}>
            打开轨迹文件
          </button>
          <button
            type="button"
            className="studio__btn studio__btn--ghost"
            onClick={() => {
              if (hasInitial) {
                setEvents(initialEvents);
                setLabel('sample-trace.jsonl');
                setError(null);
                return;
              }
              fetch(source)
                .then((response) => response.text())
                .then((text) => {
                  setEvents(parseTrace(text));
                  setLabel('sample-trace.jsonl');
                })
                .catch((reason: Error) => setError(reason.message));
            }}
          >
            载入示例
          </button>
        </div>
      </header>

      {error && <p className="studio__error">{error}</p>}

      <section className="studio__stats" aria-label="轨迹概览">
        <Stat label="run" value={run} mono />
        <Stat label="文件" value={label} mono />
        <Stat label="steps" value={String(steps.length)} />
        <Stat
          label="最终 loss"
          value={finished?.loss !== undefined ? finished.loss.toFixed(4) : '—'}
        />
        <Stat label="捕获张量" value={String(tensors.length)} />
      </section>

      <section className="studio__panel" aria-label="损失曲线">
        <PanelHead title="损失曲线" hint={`${steps.length} steps`} />
        <LossChart steps={steps} />
      </section>

      <section className="studio__panel" aria-label="注意力">
        <PanelHead
          title="注意力矩阵"
          hint={`${attention.length} 个（layer × head）`}
        />
        {attention.length === 0 ? (
          <p className="studio__empty">轨迹里没有 attention.* 张量。</p>
        ) : (
          <>
            <div className="studio__chips" role="group" aria-label="选择 head">
              {attention.map((event) => (
                <button
                  key={event.name}
                  type="button"
                  aria-pressed={selectedTensor?.name === event.name}
                  className={
                    selectedTensor?.name === event.name
                      ? 'studio__chip studio__chip--on'
                      : 'studio__chip'
                  }
                  onClick={() => setSelected(event.name)}
                >
                  {event.name.replace('attention.', '')}
                </button>
              ))}
            </div>
            {selectedTensor && <AttentionGrid tensor={selectedTensor} />}
          </>
        )}
      </section>

      <section className="studio__panel" aria-label="logits">
        <PanelHead title="下一个 token 的 logits" hint={logits ? `${logits.shape.join(' × ')}` : '—'} />
        {logits ? <LogitBars tensor={logits} /> : <p className="studio__empty">轨迹里没有 logits 张量。</p>}
      </section>

      <p className="studio__note">
        轨迹来自 <code>mini-llm-trace</code>；Studio 只读这些事件，不训练模型。token 以 id
        显示，字符表保存在 checkpoint 里。
      </p>
    </div>
  );
}

function parseTrace(text: string): TraceEvent[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as TraceEvent);
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="studio__stat">
      <span className="studio__stat-label">{label}</span>
      <span className={mono ? 'studio__stat-value mono' : 'studio__stat-value'}>{value}</span>
    </div>
  );
}

function PanelHead({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="studio__panel-head">
      <h2>{title}</h2>
      <span className="mono">{hint}</span>
    </div>
  );
}

function LossChart({ steps }: { steps: StepEvent[] }) {
  if (steps.length === 0) return <p className="studio__empty">没有 step 事件。</p>;
  const width = 720;
  const height = 180;
  const padding = { left: 36, right: 12, top: 12, bottom: 22 };
  const losses = steps.map((step) => step.loss);
  const max = Math.max(...losses);
  const min = Math.min(...losses);
  const span = max - min || 1;
  const x = (index: number) =>
    padding.left +
    (index / Math.max(1, steps.length - 1)) * (width - padding.left - padding.right);
  const y = (loss: number) =>
    padding.top + (1 - (loss - min) / span) * (height - padding.top - padding.bottom);
  const points = steps.map((step, index) => `${x(index)},${y(step.loss)}`).join(' ');
  const ticks = [max, (max + min) / 2, min];

  return (
    <svg className="studio__chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="训练损失曲线">
      {ticks.map((tick, index) => (
        <g key={index}>
          <line
            x1={padding.left}
            x2={width - padding.right}
            y1={y(tick)}
            y2={y(tick)}
            className="studio__grid"
          />
          <text x={4} y={y(tick) + 4} className="studio__axis">
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      <polyline points={points} className="studio__loss-line" />
      <circle cx={x(steps.length - 1)} cy={y(steps[steps.length - 1].loss)} r={3} className="studio__loss-dot" />
    </svg>
  );
}

function AttentionGrid({ tensor }: { tensor: TensorEvent }) {
  const [rows, cols] = tensor.shape;
  const cell = Math.max(8, Math.floor(360 / Math.max(rows, cols)));
  const size = cell * cols;
  const peak = Math.max(...tensor.values, 1e-6);

  return (
    <div className="studio__grid-wrap">
      <svg
        className="studio__heatmap"
        width={size}
        height={rows * cell}
        viewBox={`0 0 ${size} ${rows * cell}`}
        role="img"
        aria-label={`${tensor.name} 注意力热力图`}
      >
        {tensor.values.map((value, index) => {
          const row = Math.floor(index / cols);
          const col = index % cols;
          return (
            <rect
              key={index}
              x={col * cell}
              y={row * cell}
              width={cell - 1}
              height={cell - 1}
              className="studio__cell"
              opacity={Math.max(0.02, value / peak)}
            >
              <title>{`q=${row} k=${col} w=${value.toFixed(3)}`}</title>
            </rect>
          );
        })}
      </svg>
      <p className="studio__caption">
        行 = 查询位置（query），列 = 被关注位置（key）。颜色越深权重越大；上三角为因果遮罩。
      </p>
    </div>
  );
}

function LogitBars({ tensor }: { tensor: TensorEvent }) {
  const [, vocab] = tensor.shape;
  const rows = tensor.values.length / vocab;
  const last = tensor.values.slice((rows - 1) * vocab);
  const ranked = last
    .map((value, index) => ({ index, value }))
    .sort((left, right) => right.value - left.value)
    .slice(0, 10);
  const max = Math.max(...ranked.map((entry) => entry.value));
  const min = Math.min(...ranked.map((entry) => entry.value));

  return (
    <div className="studio__bars">
      {ranked.map((entry) => (
        <div key={entry.index} className="studio__bar-row">
          <span className="studio__bar-label mono">#{entry.index}</span>
          <div className="studio__bar-track">
            <div
              className="studio__bar-fill"
              style={{ width: `${((entry.value - min) / (max - min || 1)) * 100}%` }}
            />
          </div>
          <span className="studio__bar-value mono">{entry.value.toFixed(3)}</span>
        </div>
      ))}
      <p className="studio__caption">最后一个位置的 top-10 logits（按数值排序，非概率）。</p>
    </div>
  );
}
