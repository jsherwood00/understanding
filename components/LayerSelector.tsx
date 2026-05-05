"use client";

export const LAYERS = [13, 17, 21, 25, 28, 32] as const;
export type Layer = (typeof LAYERS)[number];

const LABELS: Record<Layer, string> = {
  13: "Sensory — emotional content of recent input",
  17: "Sensory–integrated",
  21: "Integrated — context being processed",
  25: "Action–integrated",
  28: "Action — preparing to express",
  32: "Output — predicting next token",
};

interface LayerSelectorProps {
  selected: Layer;
  onChange: (next: Layer) => void;
  disabled?: boolean;
}

export function LayerSelector({
  selected,
  onChange,
  disabled = false,
}: LayerSelectorProps) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-1">
        <span className="mr-1 text-[10px] tracking-[0.16em] text-ink-faint uppercase">
          layer
        </span>
        {LAYERS.map((L) => {
          const active = L === selected;
          return (
            <button
              key={L}
              type="button"
              onClick={() => onChange(L)}
              disabled={disabled}
              aria-pressed={active}
              title={LABELS[L]}
              className={
                "tabular px-1.5 py-0.5 text-[11px] transition-colors " +
                "disabled:cursor-not-allowed disabled:opacity-40 " +
                (active
                  ? "rounded border border-ink/40 text-ink"
                  : "rounded border border-transparent text-ink-faint hover:text-ink-soft")
              }
            >
              {L}
            </button>
          );
        })}
      </div>
      <span
        className="text-[10px] text-ink-faint italic"
        aria-live="polite"
      >
        {LABELS[selected]}
      </span>
    </div>
  );
}
