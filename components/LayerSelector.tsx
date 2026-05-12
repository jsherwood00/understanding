"use client";

import { LAYERS, type Layer } from "@/lib/emotions";
export { LAYERS, type Layer };

interface LayerSelectorProps {
  selected: Layer;
  onChange: (next: Layer) => void;
  disabled?: boolean;
  /** Label rendered to the left of the row (e.g. "thought", "output"). */
  label?: string;
  /** Layer to display in bold (e.g. holdout-best for the scope). Rendered
   *  bold regardless of whether it's the currently selected one. */
  best?: Layer;
}

export function LayerSelector({
  selected,
  onChange,
  disabled = false,
  label = "layer",
  best,
}: LayerSelectorProps) {
  return (
    <div className="flex min-w-0 shrink-0 items-center gap-0.5 whitespace-nowrap">
      <span className="mr-1.5 w-[3.75rem] text-[10px] tracking-[0.14em] text-ink-faint uppercase">
        {label}
      </span>
      {LAYERS.map((L) => {
        const active = L === selected;
        const isBest = L === best;
        // The best layer stays bold + dark-ink regardless of selection so
        // it's always visually highlighted as "recommended"; the selected
        // layer gets the border. Both can apply to the same button.
        const borderClass = active
          ? "border border-ink/40"
          : "border border-transparent";
        const textClass = isBest
          ? "font-bold text-ink"
          : active
            ? "text-ink"
            : "text-ink-faint hover:text-ink-soft";
        return (
          <button
            key={L}
            type="button"
            onClick={() => onChange(L)}
            disabled={disabled}
            aria-pressed={active}
            className={
              "tabular rounded px-1.5 py-0.5 text-[13px] transition-colors " +
              "disabled:cursor-not-allowed disabled:opacity-40 " +
              borderClass +
              " " +
              textClass
            }
          >
            {L}
          </button>
        );
      })}
    </div>
  );
}
