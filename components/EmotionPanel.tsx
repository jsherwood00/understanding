import {
  BEST_LAYER,
  EMOTIONS,
  EMOTION_COLORS,
  type Emotion,
  type EmotionState,
  type Turn,
} from "@/lib/emotions";
import { LayerSelector, type Layer } from "@/components/LayerSelector";
import { ViewModeToggle, type ViewMode } from "@/components/ViewModeToggle";

// Short 3-letter labels under each bar so they always fit — full names
// like SADNESS / DISGUST / SURPRISE overflow the column at any reasonable
// panel width.
const SHORT_LABEL: Record<Emotion, string> = {
  Joy: "JOY",
  Sadness: "SAD",
  Anger: "ANG",
  Fear: "FEAR",
  Disgust: "DISG",
  Surprise: "SURP",
};

interface EmotionPanelProps {
  state: EmotionState;
  turns: Turn[];
  viewingIndex: number | null;
  snapshotIndex: number;
  selectedExcerpt: string | null;
  onNavigate: (direction: -1 | 1) => void;
  onScrub: (snapIdx: number) => void;
  onReplayTurn: () => void;
  onReplayAll: () => void;
  onStopReplay: () => void;
  isReplaying: boolean;
  isGenerating: boolean;
  thoughtLayer: Layer;
  replyLayer: Layer;
  onThoughtLayerChange: (next: Layer) => void;
  onReplyLayerChange: (next: Layer) => void;
  /** When true, the dotted black NLI lines are drawn over each bar at
   *  the deberta sentiment value for that phase. */
  classifierOn: boolean;
  onClassifierToggle: (next: boolean) => void;
  /** When true, draw numeric labels next to every visible indicator
   *  (halo, reply dot, both sentiment ticks). */
  valuesOn: boolean;
  onValuesToggle: (next: boolean) => void;
  /** Three-way view switch — rendered inline next to the legend so it
   *  sits flush against the legend's right edge with no awkward gap. */
  viewMode: ViewMode;
  onViewModeChange: (next: ViewMode) => void;
}

const CLASSIFIER_NAME = "deberta-v3-large-zeroshot-v2.0";

export function EmotionPanel(props: EmotionPanelProps) {
  return (
    <section className="flex h-full min-h-0 flex-col px-10 py-8">
      <div className="flex flex-wrap items-start gap-x-3 gap-y-3">
        <div className="flex flex-col gap-3 rounded-md border border-ink-faint/25 px-3 py-2">
          <div className="flex flex-col gap-1">
            <LayerSelector
              label="thought"
              selected={props.thoughtLayer}
              onChange={props.onThoughtLayerChange}
              disabled={props.isGenerating || props.isReplaying}
              best={BEST_LAYER.thought}
            />
            <LayerSelector
              label="output"
              selected={props.replyLayer}
              onChange={props.onReplyLayerChange}
              disabled={props.isGenerating || props.isReplaying}
              best={BEST_LAYER.reply}
            />
          </div>
          <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
            <ClassifierToggle
              enabled={props.classifierOn}
              onToggle={props.onClassifierToggle}
            />
            <ValuesToggle
              enabled={props.valuesOn}
              onToggle={props.onValuesToggle}
            />
          </div>
        </div>
        <Legend classifierOn={props.classifierOn} />
        <ViewModeToggle
          mode={props.viewMode}
          onChange={props.onViewModeChange}
        />
      </div>

      <div className="mt-8 flex min-h-0 flex-1 items-stretch gap-3">
        <YAxisTitle />
        <YScale />
        {EMOTIONS.map((emotion) => (
          <Bar
            key={emotion}
            emotion={emotion}
            thinkingReply={props.state.thinkingReply[emotion]}
            thinkingThought={
              props.state.thinkingThought
                ? props.state.thinkingThought[emotion]
                : null
            }
            outputReply={
              props.state.outputReply ? props.state.outputReply[emotion] : null
            }
            outputThought={
              props.state.outputThought
                ? props.state.outputThought[emotion]
                : null
            }
            classifierOn={props.classifierOn}
            valuesOn={props.valuesOn}
            connectorOn={!props.isGenerating}
            diffOn={!props.isGenerating}
          />
        ))}
      </div>

      {props.selectedExcerpt ? (
        <ExcerptIndicator text={props.selectedExcerpt} />
      ) : (
        <TurnNavigator {...props} />
      )}
    </section>
  );
}

function ExcerptIndicator({ text }: { text: string }) {
  return (
    <div className="mt-8 border-t border-divider pt-4">
      <div className="text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        Selected excerpt
      </div>
      <div className="mt-2 line-clamp-3 text-[12px] text-ink-muted italic">
        “{text}”
      </div>
    </div>
  );
}

function ClassifierToggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: (next: boolean) => void;
}) {
  return (
    <LabeledToggle
      label={CLASSIFIER_NAME}
      enabled={enabled}
      onToggle={onToggle}
    />
  );
}

function ValuesToggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: (next: boolean) => void;
}) {
  return (
    <LabeledToggle
      label="show values"
      enabled={enabled}
      onToggle={onToggle}
    />
  );
}

function LabeledToggle({
  label,
  enabled,
  onToggle,
}: {
  label: string;
  enabled: boolean;
  onToggle: (next: boolean) => void;
}) {
  return (
    <div className="flex min-w-0 shrink-0 flex-col gap-1">
      <span className="text-[10px] tracking-[0.14em] text-ink-faint uppercase">
        {label}
      </span>
      <button
        type="button"
        onClick={() => onToggle(!enabled)}
        aria-pressed={enabled}
        className={
          "flex items-center gap-2 self-start rounded-full border px-2 py-0.5 text-[12px] tracking-wide transition-colors " +
          (enabled
            ? "border-ink/40 text-ink"
            : "border-ink/15 text-ink-faint hover:text-ink-soft")
        }
      >
        <span
          className={
            "block h-2.5 w-2.5 rounded-full " +
            (enabled ? "bg-ink" : "bg-ink-faint/40")
          }
        />
        <span>{enabled ? "on" : "off"}</span>
      </button>
    </div>
  );
}

// Chart inset — keep 18 px of padding at the top and bottom of the chart
// area so a 36-px halo/dot centered at value=0 or value=100 fits inside
// the visible area (centered on the 0 / 100 axis line, not clipped or
// pushed up off-axis). All positioned elements — y-axis ticks, halos,
// dots, sentiment ticks, value labels — go through chartBottom() so they
// share the same coordinate system.
const CHART_PAD_PX = 18;

/** CSS `bottom` for an indicator whose CENTER should sit at `pct%` on the
 *  insetted chart axis. `heightPx` is the indicator's full height (used
 *  to convert center → bottom). 0% maps to CHART_PAD_PX from the bottom
 *  of the container; 100% maps to CHART_PAD_PX from the top. */
function chartBottom(pct: number, heightPx: number): string {
  return `calc(${pct} / 100 * (100% - ${CHART_PAD_PX * 2}px) + ${CHART_PAD_PX - heightPx / 2}px)`;
}

function Bar({
  emotion,
  thinkingReply,
  thinkingThought,
  outputReply,
  outputThought,
  classifierOn,
  valuesOn,
  connectorOn,
  diffOn,
}: {
  emotion: Emotion;
  /** Filled solid dot — activation projection averaged over reply tokens. */
  thinkingReply: number;
  /** Halo (cloud) — activation projection averaged over thought tokens.
   *  Null when the model didn't emit a thought block. */
  thinkingThought: number | null;
  /** Hollow black tick — NLI sentiment of the reply text. Null mid-stream.
   *  Only drawn when classifierOn is true. */
  outputReply: number | null;
  /** Fuzzy black tick — NLI sentiment of the thought block. Null mid-
   *  stream or when there's no thought block. Only drawn when
   *  classifierOn is true. */
  outputThought: number | null;
  /** Master switch for the NLI ticks. */
  classifierOn: boolean;
  /** When true, render numeric labels next to every visible indicator. */
  valuesOn: boolean;
  /** When false, suppress the halo-to-dot connector. The connector is
   *  hidden during live streaming because both endpoints are still
   *  moving and the line reads as chaos. Re-enabled at end-of-turn and
   *  during replay. */
  connectorOn: boolean;
  /** When false, suppress the "thought − reply" diff label below the
   *  bar. Hidden during the thought-phase prefix of a streaming turn
   *  (no reply tokens yet → diff is just −thought, not meaningful). */
  diffOn: boolean;
}) {
  const color = EMOTION_COLORS[emotion];
  const haloPct = thinkingThought !== null ? clamp(thinkingThought) : 0;
  const dotPct = clamp(thinkingReply);
  const thoughtNliPct = outputThought !== null ? clamp(outputThought) : 0;
  const replyNliPct = outputReply !== null ? clamp(outputReply) : 0;
  const haloColor = `color-mix(in srgb, ${color} 70%, #faf9f6)`;
  const ringColor = `color-mix(in srgb, ${color} 80%, #1a1a1a)`;
  const connectorColor = `color-mix(in srgb, ${color} 60%, #1a1a1a)`;
  // Connecting bar between the halo (thought) and hollow dot (reply) —
  // visualizes the shift between phases. Anchored at whichever value is
  // lower, with height equal to the absolute gap.
  const connectorLo = Math.min(haloPct, dotPct);
  const connectorHi = Math.max(haloPct, dotPct);
  const connectorSpan = connectorHi - connectorLo;
  // Connector requires BOTH endpoints to be real AND the caller-supplied
  // gate (off during live streaming, on after generation / during replay).
  const showConnector =
    connectorOn &&
    thinkingThought !== null &&
    haloPct > 0 &&
    dotPct > 0 &&
    connectorSpan > 0;
  // Difference is signed: thought − output. Positive = thought over reply
  // (model felt this more strongly internally than it expressed), negative
  // = reply over thought (model expressed more than it "felt"). If no
  // thought block, treat thought as 0.
  const diff = Math.round(haloPct) - Math.round(dotPct);
  const diffLabel = diff > 0 ? `+${diff}` : diff < 0 ? `${diff}` : "0";

  return (
    <div className="flex min-w-0 flex-1 flex-col items-center">
      <div className="relative my-3 w-full max-w-[44px] flex-1">
        {/* Connector — thin vertical line bridging the halo and reply dot
            EDGE-to-EDGE (not center-to-center), so it doesn't pass through
            either circle. Radius is 18px → trim 18px from each end. CSS
            `max(0, …)` collapses the line to nothing when the circles
            overlap. */}
        {showConnector && (
          <div
            className="absolute left-1/2 -translate-x-1/2"
            style={{
              bottom: `calc(${connectorLo} / 100 * (100% - ${CHART_PAD_PX * 2}px) + ${CHART_PAD_PX * 2}px)`,
              height: `max(0px, calc(${connectorSpan} / 100 * (100% - ${CHART_PAD_PX * 2}px) - ${CHART_PAD_PX * 2}px))`,
              width: "2px",
              backgroundColor: connectorColor,
              opacity: 0.45,
              transition: "bottom 300ms ease-out, height 300ms ease-out",
            }}
          />
        )}
        {/* Halo (thought activations) — soft blurry filled circle. The
            same clamp pattern as ticks: pins to the floor at value=0
            (bottom at 0px, top at +36px) and to the ceiling at value=100
            so the disc never reads as "above 0" or "above 100". */}
        {thinkingThought !== null && haloPct > 0 && (
          <div
            className="absolute left-1/2 -translate-x-1/2 rounded-full"
            style={{
              bottom: chartBottom(haloPct, 36),
              width: "36px",
              height: "36px",
              backgroundColor: haloColor,
              opacity: 0.5,
              filter: "blur(3px)",
              transition: "bottom 300ms ease-out",
            }}
          />
        )}
        {/* Filled dot (reply activations) — same size as halo, sharp solid
            fill in the darker emotion shade. Same clamp pattern. */}
        {dotPct > 0 && (
          <div
            className="absolute left-1/2 -translate-x-1/2 rounded-full"
            style={{
              bottom: chartBottom(dotPct, 36),
              width: "36px",
              height: "36px",
              backgroundColor: ringColor,
              transition: "bottom 300ms ease-out",
            }}
          />
        )}
        {/* Fuzzy black tick — thought NLI. Always rendered when the
            classifier has produced a reading (even at value 0); we want
            all 6 ticks visible so the user can see "this emotion was
            checked and came back at the floor," not "this emotion
            disappeared." Wider than the 36px dot so the edges always
            poke out past whichever dot/halo sits at the same height. */}
        {classifierOn && outputThought !== null && (
          <div
            className="peer/thought-tick absolute left-1/2 -translate-x-1/2"
            style={{
              bottom: chartBottom(thoughtNliPct, 6),
              width: "56px",
              height: "6px",
              backgroundColor: "#1a1a1a",
              borderRadius: "3px",
              opacity: 0.55,
              filter: "blur(2px)",
              zIndex: 3,
              transition: "bottom 300ms ease-out",
            }}
          />
        )}
        {/* Hollow black tick — reply NLI. */}
        {classifierOn && outputReply !== null && (
          <div
            className="peer/reply-tick absolute left-1/2 -translate-x-1/2"
            style={{
              bottom: chartBottom(replyNliPct, 4),
              width: "56px",
              height: "4px",
              border: "1.5px solid #1a1a1a",
              background: "transparent",
              zIndex: 3,
              transition: "bottom 300ms ease-out",
            }}
          />
        )}
        {/* Numeric labels — drawn next to each visible indicator when
            valuesOn is true. Thought-side values go to the LEFT of the
            column; reply-side go to the RIGHT. */}
        {valuesOn && thinkingThought !== null && haloPct > 0 && (
          <ValueLabel side="left" pct={haloPct} value={haloPct} />
        )}
        {valuesOn && dotPct > 0 && (
          <ValueLabel
            side="right"
            pct={dotPct}
            value={dotPct}
            color="#faf9f6"
            background={ringColor}
          />
        )}
        {/* Tick labels are hover-only, independent of the values toggle:
            they appear on mouseover of the peer tick element. */}
        {classifierOn && outputThought !== null && (
          <ValueLabel
            side={pickTickSide({
              tickPct: thoughtNliPct,
              defaultSide: "left",
              leftNeighborPct: thinkingThought !== null && haloPct > 0 ? haloPct : null,
              rightNeighborPct: dotPct > 0 ? dotPct : null,
            })}
            pct={thoughtNliPct}
            value={thoughtNliPct}
            dim
            clearTick
            hoverPeer="thought-tick"
          />
        )}
        {classifierOn && outputReply !== null && (
          <ValueLabel
            side={pickTickSide({
              tickPct: replyNliPct,
              defaultSide: "right",
              leftNeighborPct: thinkingThought !== null && haloPct > 0 ? haloPct : null,
              rightNeighborPct: dotPct > 0 ? dotPct : null,
            })}
            pct={replyNliPct}
            value={replyNliPct}
            dim
            clearTick
            hoverPeer="reply-tick"
          />
        )}
      </div>
      <span
        className="smallcaps block w-full text-center font-bold text-ink"
        title={emotion}
      >
        {SHORT_LABEL[emotion]}
      </span>
      <span
        className="tabular block w-full text-center text-[11px] text-ink-soft"
        title="thought activations − reply activations"
      >
        {diffOn ? diffLabel : " "}
      </span>
    </div>
  );
}


// Vertical distance (percentage points) below which a tick label is
// considered to be at the same height as a neighboring indicator and
// will visually clash. Tuned by eye against the 36-px halo/dot diameter.
const TICK_LABEL_CLASH_PCT = 10;

/**
 * Pick which side of the column a tick label should sit on.
 *
 * Defaults: thought-NLI label → left, reply-NLI label → right. Flip
 * whenever the activation indicator on the default side would be within
 * TICK_LABEL_CLASH_PCT of the tick. If BOTH sides would clash, fall
 * back to whichever side has more clearance.
 */
function pickTickSide({
  tickPct,
  defaultSide,
  leftNeighborPct,
  rightNeighborPct,
}: {
  tickPct: number;
  defaultSide: "left" | "right";
  /** Vertical position of the indicator on the LEFT side (the halo's
   *  activation value's label) — null if no indicator is rendered there. */
  leftNeighborPct: number | null;
  /** Vertical position of the indicator on the RIGHT side (the dot's
   *  in-circle chip) — null if no indicator is rendered there. */
  rightNeighborPct: number | null;
}): "left" | "right" {
  const leftDist =
    leftNeighborPct === null ? Infinity : Math.abs(tickPct - leftNeighborPct);
  const rightDist =
    rightNeighborPct === null ? Infinity : Math.abs(tickPct - rightNeighborPct);
  const leftClear = leftDist >= TICK_LABEL_CLASH_PCT;
  const rightClear = rightDist >= TICK_LABEL_CLASH_PCT;

  if (defaultSide === "left" && leftClear) return "left";
  if (defaultSide === "right" && rightClear) return "right";
  // Default side has a clash. Try the other side.
  if (defaultSide === "left" && rightClear) return "right";
  if (defaultSide === "right" && leftClear) return "left";
  // Both sides clash: pick the one with more clearance.
  return rightDist > leftDist ? "right" : "left";
}

function ValueLabel({
  side,
  pct,
  value,
  dim = false,
  color,
  background,
  clearTick = false,
  hoverPeer,
}: {
  side: "left" | "right";
  /** Vertical position as a percentage of column height. */
  pct: number;
  /** Numeric value to display (rounded to integer). */
  value: number;
  /** Render in a muted color (used for sentiment ticks). */
  dim?: boolean;
  /** Optional override text color (used to land readable text inside
   *  the filled reply dot). */
  color?: string;
  /** Optional background fill (for labels sitting inside a dot). */
  background?: string;
  /** When true, push the label out far enough to clear the 56-px-wide
   *  sentiment tick (vs. the 36-px-wide halo/dot). Set on tick labels. */
  clearTick?: boolean;
  /** When set, this label is hidden by default and only shown while the
   *  user hovers the matching `peer/{name}` tick element earlier in the
   *  same parent. Used for the sentiment-tick labels. */
  hoverPeer?: "thought-tick" | "reply-tick";
}) {
  const offsetPx = clearTick ? 34 : 22;
  const positionStyle =
    side === "left"
      ? { right: `calc(50% + ${offsetPx}px)` }
      : { left: `calc(50% + ${offsetPx}px)` };
  const hoverClass =
    hoverPeer === "thought-tick"
      ? " opacity-0 peer-hover/thought-tick:opacity-100"
      : hoverPeer === "reply-tick"
        ? " opacity-0 peer-hover/reply-tick:opacity-100"
        : "";
  return (
    <span
      className={
        "tabular pointer-events-none absolute text-[10px] font-medium whitespace-nowrap transition-opacity duration-150" +
        hoverClass
      }
      style={{
        bottom: chartBottom(pct, 14),
        ...positionStyle,
        color: color ?? (dim ? "#6e6a62" : "#2a2823"),
        backgroundColor: background ?? "rgba(250, 249, 246, 0.85)",
        padding: "0 3px",
        borderRadius: "3px",
        zIndex: 4,
        transition: "bottom 300ms ease-out, opacity 150ms ease-out",
      }}
    >
      {Math.round(value)}
    </span>
  );
}

// Static y-scale to the left of the bars. Major ticks at 25/50/75/100
// with a small numeric label; minor ticks every 5 in between, unlabeled.
// The scale is purely a visual reference — it doesn't depend on state.
function YAxisTitle() {
  return (
    <div className="flex w-4 shrink-0 items-center justify-center">
      <span
        className="smallcaps text-[10px] tracking-[0.16em] whitespace-nowrap text-ink-faint"
        style={{
          writingMode: "vertical-rl",
          transform: "rotate(180deg)",
        }}
      >
        Emotion intensity (percentile)
      </span>
    </div>
  );
}

function YScale() {
  const MAJORS = [0, 25, 50, 75, 100];
  const MINORS: number[] = [];
  for (let v = 5; v <= 100; v += 5) {
    if (!MAJORS.includes(v)) MINORS.push(v);
  }
  return (
    <div className="flex w-7 shrink-0 flex-col">
      <div className="relative my-3 flex-1">
        {MINORS.map((v) => (
          <div
            key={`min-${v}`}
            className="absolute right-0 bg-ink-faint/25"
            style={{
              bottom: chartBottom(v, 1),
              width: "3px",
              height: "1px",
            }}
          />
        ))}
        {MAJORS.map((v) => (
          <div
            key={`maj-${v}`}
            className="absolute right-0 flex translate-y-1/2 items-center gap-1"
            style={{ bottom: chartBottom(v, 0) }}
          >
            <span className="tabular text-[9px] text-ink-faint">{v}</span>
            <span
              className="block bg-ink-faint/50"
              style={{ width: "5px", height: "1px" }}
            />
          </div>
        ))}
      </div>
      {/* Bottom spacer matches the bars' emotion-label row so the
          chart-area top/bottom of the scale lines up with the bars'. */}
      {/* Two hidden spacer rows that exactly mirror the emotion-label
          and diff-label rows in each Bar — without these the YScale's
          chart-area `flex-1` ends up taller than the Bar's chart-area,
          so 100% in chartBottom resolves to different pixel heights and
          the y-axis ticks drift away from the data ticks. */}
      <span
        className="smallcaps block w-full text-center font-bold"
        aria-hidden
      >
        &nbsp;
      </span>
      <span
        className="tabular block w-full text-center text-[11px]"
        aria-hidden
      >
        &nbsp;
      </span>
    </div>
  );
}

function Legend({ classifierOn }: { classifierOn: boolean }) {
  return (
    <div className="flex shrink-0 flex-col items-start gap-1.5 whitespace-nowrap rounded-md border border-ink-faint/25 px-3 py-2 text-[14px] text-ink-soft">
      <span className="flex items-center gap-2">
        <span
          className="block h-4 w-4 shrink-0 rounded-full"
          style={{
            background:
              "radial-gradient(circle, #4a4a4a 0%, #4a4a4a55 40%, transparent 70%)",
            filter: "blur(1.5px)",
          }}
        />
        thought activations
      </span>
      <span className="flex items-center gap-2">
        <span
          className="block h-3.5 w-3.5 shrink-0 rounded-full"
          style={{ backgroundColor: "#4a4a4a" }}
        />
        reply activations
      </span>
      {classifierOn && (
        <>
          <span className="flex items-center gap-2">
            <span
              className="block h-[6px] w-5 shrink-0"
              style={{
                backgroundColor: "#1a1a1a",
                borderRadius: "3px",
                opacity: 0.55,
                filter: "blur(2px)",
              }}
            />
            thought sentiment
          </span>
          <span className="flex items-center gap-2">
            <span
              className="block h-[3px] w-5 shrink-0"
              style={{ border: "1px solid #1a1a1a", background: "transparent" }}
            />
            reply sentiment
          </span>
        </>
      )}
    </div>
  );
}

// Snapshots are pushed every N tokens during generation. Must match
// SNAPSHOT_EVERY_N_TOKENS in Workspace.tsx — the relationship is:
// snapshot index k corresponds to tokens [N*k .. N*(k+1)-1] (the chunk
// generated since the previous snapshot). Snapshot k's `thinking` value
// is the residual-stream reading at the *last* of those N tokens; the
// chunk is what the user is "looking at" at this scrub position.
const SNAPSHOT_EVERY_N_TOKENS = 5;

/** When the user scrubs to a non-final snapshot, return the literal output
 *  text generated during that snapshot's chunk (N tokens). At the final
 *  snapshot (which represents the turn-wide average, not a single chunk),
 *  return null so the caller falls back to the user prompt preview.
 *  Per-token charEnd is in *phase-buffer* coordinates, so we slice from
 *  the buffer that matches each token's phase. If the chunk straddles a
 *  phase boundary we just return whatever fits in the last token's buffer. */
function chunkTextForSnapshot(turn: Turn, snapIdx: number): string | null {
  const lastIdx = turn.snapshots.length - 1;
  if (snapIdx >= lastIdx) return null;
  const tokenEnd = (snapIdx + 1) * SNAPSHOT_EVERY_N_TOKENS - 1;
  if (tokenEnd >= turn.tokens.length) return null;
  const tokenStart = snapIdx * SNAPSHOT_EVERY_N_TOKENS;
  const endTok = turn.tokens[tokenEnd];
  const startTok = turn.tokens[tokenStart];
  const buffer =
    endTok.phase === "thought" ? turn.assistantThought : turn.assistantReply;
  const charStart =
    tokenStart > 0 && turn.tokens[tokenStart - 1].phase === endTok.phase
      ? turn.tokens[tokenStart - 1].charEnd
      : startTok.phase === endTok.phase
        ? 0
        : 0;
  return buffer.slice(charStart, endTok.charEnd);
}

function TurnNavigator(props: Omit<EmotionPanelProps, "state">) {
  const turns = props.turns ?? [];
  const {
    viewingIndex,
    snapshotIndex,
    onNavigate,
    onScrub,
    onReplayTurn,
    onReplayAll,
    onStopReplay,
    isReplaying,
    isGenerating,
  } = props;

  if (turns.length === 0) return <div className="mt-8 h-12" />;

  const turn = viewingIndex !== null ? turns[viewingIndex] : null;
  const snapCount = turn?.snapshots.length ?? 0;
  const sliderMax = Math.max(0, snapCount - 1);
  const isAtStart = viewingIndex === null || viewingIndex === 0;
  const isAtEnd =
    viewingIndex === null || viewingIndex === turns.length - 1;
  const navDisabled = isReplaying || isGenerating;
  const scrubDisabled = navDisabled || snapCount <= 1;

  // At a non-final scrub position: show the output chunk being analyzed
  // at this snapshot. Otherwise show the user prompt preview as before.
  const chunkText = turn ? chunkTextForSnapshot(turn, snapshotIndex) : null;
  const showingChunk = chunkText !== null;
  const previewText = showingChunk
    ? chunkText.replace(/\s+/g, " ").trim()
    : turn
      ? previewWords(turn.userMessage, 10)
      : "";
  const previewLabel = showingChunk
    ? `tokens ${snapshotIndex * SNAPSHOT_EVERY_N_TOKENS + 1}–${(snapshotIndex + 1) * SNAPSHOT_EVERY_N_TOKENS}`
    : "prompt";

  return (
    <div className="mt-8 border-t border-divider pt-4">
      <div className="mb-3 flex items-baseline gap-2">
        <span className="smallcaps shrink-0 text-[10px] text-ink-faint">
          {previewLabel}
        </span>
        <span className="truncate text-[12px] text-ink-muted italic">
          {previewText ? `"${previewText}"` : "—"}
        </span>
      </div>

      <div className="flex items-center gap-3 text-xs">
        {/* Snapshot scrubber on the left, takes the bulk of the row */}
        <div className="flex flex-1 items-center gap-2">
          <input
            type="range"
            min={0}
            max={sliderMax}
            value={Math.min(snapshotIndex, sliderMax)}
            onChange={(e) => onScrub(Number(e.target.value))}
            disabled={scrubDisabled}
            className="h-1 flex-1 cursor-pointer accent-ink-soft disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Scrub through chunks of this turn"
          />
          <span className="tabular text-[10px] text-ink-faint">
            {Math.min(snapshotIndex, sliderMax) + 1}/{Math.max(1, snapCount)}
          </span>
        </div>

        {/* Turn navigator on the right */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => onNavigate(-1)}
            disabled={navDisabled || isAtStart}
            className="rounded px-1.5 py-0.5 text-ink-soft hover:bg-tint disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:bg-transparent"
            aria-label="Previous turn"
          >
            ‹
          </button>
          <span className="tabular text-ink-muted">
            {viewingIndex !== null ? viewingIndex + 1 : 0}/{turns.length}
          </span>
          <button
            type="button"
            onClick={() => onNavigate(1)}
            disabled={navDisabled || isAtEnd}
            className="rounded px-1.5 py-0.5 text-ink-soft hover:bg-tint disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:bg-transparent"
            aria-label="Next turn"
          >
            ›
          </button>
        </div>
      </div>

      {/* Replay actions */}
      <div className="mt-2 flex justify-end gap-3 text-[11px] tracking-wide text-ink-muted uppercase">
        {isReplaying ? (
          <button
            type="button"
            onClick={onStopReplay}
            className="rounded border border-ink/30 px-2 py-0.5 text-ink-soft hover:border-ink hover:text-ink"
          >
            stop
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={onReplayTurn}
              disabled={isGenerating || viewingIndex === null}
              className="hover:text-ink-soft disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:text-ink-faint"
            >
              replay
            </button>
            <button
              type="button"
              onClick={onReplayAll}
              disabled={isGenerating || turns.length === 0}
              className="hover:text-ink-soft disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:text-ink-faint"
            >
              replay all
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function previewWords(text: string, maxWords: number): string {
  const words = text.trim().split(/\s+/);
  if (words.length <= maxWords) return text.trim();
  return words.slice(0, maxWords).join(" ") + "…";
}

function clamp(n: number) {
  return Math.max(0, Math.min(100, n));
}
