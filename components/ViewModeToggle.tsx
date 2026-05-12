"use client";

export type ViewMode = "bars" | "thought_map" | "output_map";

const ENTRIES: readonly (readonly [ViewMode, string])[] = [
  ["bars", "bars"],
  ["thought_map", "thought map"],
  ["output_map", "output map"],
] as const;

interface ViewModeToggleProps {
  mode: ViewMode;
  onChange: (next: ViewMode) => void;
}

/** Vertical 3-way switch for the left panel: bars / thought map / output
 *  map. Drops into a flex row next to the legend (or anywhere else); does
 *  not absolutely position itself. */
export function ViewModeToggle({ mode, onChange }: ViewModeToggleProps) {
  return (
    <div className="flex shrink-0 flex-col gap-1 rounded-md border border-ink-faint/30 bg-canvas/85 px-1.5 py-2 text-[11px] tracking-wide backdrop-blur-sm">
      {ENTRIES.map(([m, label]) => (
        <button
          key={m}
          type="button"
          onClick={() => onChange(m)}
          aria-pressed={mode === m}
          className={
            "whitespace-nowrap rounded px-2 py-1 text-left transition-colors " +
            (mode === m
              ? "bg-ink/10 text-ink"
              : "text-ink-faint hover:text-ink-soft")
          }
        >
          {label}
        </button>
      ))}
    </div>
  );
}
