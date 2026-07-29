import wordmark from "../assets/exozippy_wordmark.png";

// Top bar: wordmark left, project name center, global actions right. Actions
// are placeholders in G7 -- later prompts (G8 save/undo, G10 Solve, G11 Run)
// wire them up.
export default function TopBar({ projectName }: { projectName: string | null }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <img className="wordmark" src={wordmark} alt="EXOZIPPy" />
      </div>
      <div className="topbar-center">
        {projectName ? (
          <span className="project-name">{projectName}</span>
        ) : (
          <span className="project-name muted">no project open</span>
        )}
      </div>
      <div className="topbar-right">
        <span className="badge" title="These become active in later prompts">
          shell
        </span>
      </div>
    </header>
  );
}
