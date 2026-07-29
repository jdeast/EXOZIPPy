import { useEffect, useRef, useState } from "react";
import { openLogSocket } from "../api";

// Bottom terminal panel: streams a log file over the /api/logs WebSocket with
// autoscroll and a pause toggle. Collapsible. G11 will point this at a running
// fit's log file; here it tails whatever file the sidebar selects.
export default function LogTerminal({ file }: { file: string | null }) {
  const [lines, setLines] = useState<string[]>([]);
  const [autoscroll, setAutoscroll] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const [connected, setConnected] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    setLines([]);
    if (!file) {
      setConnected(false);
      return;
    }
    const ws = openLogSocket(file);
    socketRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (ev) => {
      // Cap retained lines so a long run does not grow the DOM unbounded.
      setLines((prev) => {
        const next = prev.concat(ev.data as string);
        return next.length > 5000 ? next.slice(next.length - 5000) : next;
      });
    };
    return () => ws.close();
  }, [file]);

  useEffect(() => {
    if (autoscroll && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [lines, autoscroll]);

  return (
    <div className={`terminal ${collapsed ? "collapsed" : ""}`}>
      <div className="terminal-header">
        <button className="terminal-toggle" onClick={() => setCollapsed((c) => !c)}>
          {collapsed ? "▲" : "▼"} Log
        </button>
        <span className="terminal-file" title={file || ""}>
          {file ? file.split("/").pop() : "no file selected"}
        </span>
        <span className={`terminal-status ${connected ? "on" : "off"}`}>
          {file ? (connected ? "streaming" : "waiting") : "idle"}
        </span>
        <label className="terminal-autoscroll">
          <input
            type="checkbox"
            checked={autoscroll}
            onChange={(e) => setAutoscroll(e.target.checked)}
          />
          autoscroll
        </label>
        <button className="terminal-clear" onClick={() => setLines([])}>
          clear
        </button>
      </div>
      {!collapsed && (
        <div className="terminal-body" ref={bodyRef}>
          {lines.map((l, i) => (
            <div className="terminal-line" key={i}>
              {l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
