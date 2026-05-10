// /monitor — overnight-run babysitter dashboard.
//
// This is a Server Component shell only. Everything live (GPU stats,
// log tail, storage, pause toggle) is in MonitorClient.tsx, which
// owns the polling. The page is intentionally cheap to render so the
// chat UI on /  isn't affected by traffic here.

import { MonitorClient } from "@/components/MonitorClient";

export const dynamic = "force-dynamic";

export default function MonitorPage() {
  return (
    <main className="min-h-full overflow-auto bg-canvas px-8 py-8">
      <div className="mx-auto flex max-w-4xl flex-col gap-6">
        <header className="flex items-baseline justify-between">
          <h1 className="font-serif text-[26px] tracking-tight text-ink">
            pipeline monitor
          </h1>
          <span className="smallcaps text-ink-faint">overnight runs</span>
        </header>
        <MonitorClient />
      </div>
    </main>
  );
}
