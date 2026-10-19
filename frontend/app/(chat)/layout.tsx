import { ChatShell } from "@/components/ChatShell";

/** The shell stays mounted across `/` and `/c/[id]`, so starting a chat does not reset it. */
export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <ChatShell />
      {children}
    </>
  );
}
