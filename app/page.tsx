import { Header } from "@/components/Header";
import { Workspace } from "@/components/Workspace";

export default function Home() {
  return (
    <div className="flex h-full flex-col">
      <Header />
      <main className="flex min-h-0 flex-1">
        <Workspace />
      </main>
    </div>
  );
}
