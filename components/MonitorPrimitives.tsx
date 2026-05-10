"use client";

// Shared primitives across the four monitor cards. Pulled out so
// every card has the same border treatment and label cadence
// without each file restating the Tailwind classes.

import type { ReactNode } from "react";

export function Card({ children }: { children: ReactNode }) {
  return (
    <section className="rounded-md border border-divider bg-canvas px-5 py-4">
      {children}
    </section>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return <h2 className="smallcaps text-ink-muted">{children}</h2>;
}
