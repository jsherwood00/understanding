import {
  EMOTIONS,
  EMOTION_COLORS,
  type Emotion,
  type EmotionState,
  type Turn,
} from "@/lib/emotions";
import { LayerSelector, type Layer } from "@/components/LayerSelector";

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
  selectedLayer: Layer;
  onLayerChange: (next: Layer) => void;
}

export function EmotionPanel(props: EmotionPanelProps) {
  return (
    <section className="flex h-full min-h-0 flex-col px-10 py-8">
      <div className="flex flex-col gap-3">
        <LayerSelector
          selected={props.selectedLayer}
          onChange={props.onLayerChange}
          disabled={props.isGenerating || props.isReplaying}
        />
        <div className="flex justify-end">
          <Legend />
        </div>
      </div>

      <div className="mt-8 flex min-h-0 flex-1 items-stretch gap-3">
        <YScale />
        {EMOTIONS.map((emotion) => (
          <Bar
            key={emotion}
            emotion={emotion}
            thinking={props.state.thinking[emotion]}
            output={
              props.state.output ? props.state.output[emotion] : null
            }
            showNumbers={!props.isGenerating}
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

function Bar({
  emotion,
  thinking,
  output,
  showNumbers,
}: {
  emotion: Emotion;
  thinking: number;
  /** null while a turn is mid-stream — the dot hides until distilroberta
   *  classifies the full reply at end-of-turn. */
  output: number | null;
  /** When false (during streaming), the readout numbers are hidden so the
   *  user watches the halos move; numbers come back at rest, when the
   *  scrubber is the way to inspect the trace. */
  showNumbers: boolean;
}) {
  const color = EMOTION_COLORS[emotion];
  const thinkingPct = clamp(thinking);
  const outputPct = output !== null ? clamp(output) : 0;
  const dotColor = `color-mix(in srgb, ${color} 80%, #1a1a1a)`;
  const haloColor = `color-mix(in srgb, ${color} 70%, #faf9f6)`;

  return (
    <div className="flex min-w-0 flex-1 flex-col items-center">
      <div className="relative my-3 w-full max-w-[44px] flex-1 rounded-sm">
        {/* Thinking — soft halo, much larger than the dot so it stays visible
            even when values overlap. Solid lighter color, no blur. */}
        <div
          className="absolute left-1/2 -translate-x-1/2 rounded-full"
          style={{
            bottom: `calc(${thinkingPct}% - 18px)`,
            width: "36px",
            height: "36px",
            backgroundColor: haloColor,
            opacity: 0.5,
            filter: "blur(3px)",
            transition: "bottom 300ms ease-out",
          }}
        />
        {/* Thinking value — centered on the halo, on top of it. Hidden
            during streaming; appears when the scrubber is in play. */}
        {showNumbers && (
          <span
            className="tabular absolute left-1/2 -translate-x-1/2 text-[11px] font-semibold text-ink"
            style={{
              bottom: `calc(${thinkingPct}% - 7px)`,
              transition: "bottom 300ms ease-out",
            }}
          >
            {Math.round(thinking)}
          </span>
        )}
        {/* Output — sharp solid dot, only rendered once distilroberta has
            classified the full reply at end-of-turn. */}
        {output !== null && (
          <>
            <div
              className="absolute left-1/2 -translate-x-1/2 rounded-full"
              style={{
                bottom: `calc(${outputPct}% - 8px)`,
                width: "16px",
                height: "16px",
                backgroundColor: dotColor,
                transition: "bottom 300ms ease-out",
              }}
            />
            {/* Output value — small label just to the right of the dot.
                Hidden during streaming alongside the thinking label. */}
            {showNumbers && (
              <span
                className="tabular absolute text-[12px] text-ink-soft"
                style={{
                  bottom: `calc(${outputPct}% - 7px)`,
                  left: "calc(50% + 12px)",
                  transition: "bottom 300ms ease-out",
                }}
              >
                {Math.round(output)}
              </span>
            )}
          </>
        )}
      </div>
      <span
        className="smallcaps block w-full text-center text-ink-muted"
        title={emotion}
      >
        {SHORT_LABEL[emotion]}
      </span>
    </div>
  );
}


// Static y-scale to the left of the bars. Major ticks at 25/50/75/100
// with a small numeric label; minor ticks every 5 in between, unlabeled.
// The scale is purely a visual reference — it doesn't depend on state.
function YScale() {
  const MAJORS = [25, 50, 75, 100];
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
              bottom: `${v}%`,
              width: "3px",
              height: "1px",
            }}
          />
        ))}
        {MAJORS.map((v) => (
          <div
            key={`maj-${v}`}
            className="absolute right-0 flex translate-y-1/2 items-center gap-1"
            style={{ bottom: `${v}%` }}
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
      <span className="smallcaps block w-full text-center" aria-hidden>
        &nbsp;
      </span>
    </div>
  );
}

function Legend() {
  return (
    <div className="flex shrink-0 items-center gap-4 text-[13px] whitespace-nowrap text-ink-soft">
      <span className="flex items-center gap-2">
        <span className="h-3.5 w-3.5 shrink-0 rounded-full bg-ink-soft" />
        Output text
      </span>
      <span className="flex items-center gap-2">
        <span
          className="h-5 w-5 shrink-0 rounded-full"
          style={{
            background:
              "radial-gradient(circle, #4a4a4a 0%, #4a4a4a55 40%, transparent 70%)",
            filter: "blur(1.5px)",
          }}
        />
        Internal state
      </span>
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
 *  return null so the caller falls back to the user prompt preview. */
function chunkTextForSnapshot(turn: Turn, snapIdx: number): string | null {
  const lastIdx = turn.snapshots.length - 1;
  if (snapIdx >= lastIdx) return null;
  const tokenEnd = (snapIdx + 1) * SNAPSHOT_EVERY_N_TOKENS - 1;
  if (tokenEnd >= turn.tokens.length) return null;
  const tokenStart = snapIdx * SNAPSHOT_EVERY_N_TOKENS;
  const charStart =
    tokenStart > 0 ? turn.tokens[tokenStart - 1].charEnd : 0;
  const charEnd = turn.tokens[tokenEnd].charEnd;
  return turn.assistantReply.slice(charStart, charEnd);
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
