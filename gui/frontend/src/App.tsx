import { useEffect, useState } from "react";
import TopBar from "./components/TopBar";
import Sidebar from "./components/Sidebar";
import LogTerminal from "./components/LogTerminal";
import WelcomeTab from "./components/WelcomeTab";
import ConfigTab from "./components/ConfigTab";
import DataTab from "./components/DataTab";
import TuneTab from "./components/TuneTab";
import RunTab from "./components/RunTab";
import ToolsTab from "./components/ToolsTab";
import { api, type ProjectListing, type FileEntry } from "./api";

// Application shell: top bar + left sidebar + center tabbed workspace + bottom
// log terminal. G7 shipped Welcome; G8 adds Config. Later prompts register
// Data, Tune, Run, Canvas, and Results tabs into the same workspace.

// Shared context each tab's render receives, so a tab can read the open project,
// attach the bottom terminal to a log file it cares about, and (Config tab) edit
// the project's config file.
interface TabContext {
  listing: ProjectListing | null;
  setLogFile: (file: string | null) => void;
  configPath: string | null;
  setActiveTab: (id: string) => void;
}

interface Tab {
  id: string;
  label: string;
  render: (ctx: TabContext) => JSX.Element;
}

const TABS: Tab[] = [
  {
    id: "welcome",
    label: "Welcome",
    render: (ctx) => (
      <WelcomeTab configPath={ctx.configPath} setActiveTab={ctx.setActiveTab} />
    ),
  },
  { id: "config", label: "Config", render: (ctx) => <ConfigTab configPath={ctx.configPath} /> },
  { id: "tune", label: "Tune", render: (ctx) => <TuneTab configPath={ctx.configPath} /> },
  {
    id: "data",
    label: "Data",
    render: (ctx) => <DataTab listing={ctx.listing} configPath={ctx.configPath} />,
  },
  {
    id: "run",
    label: "Run",
    render: (ctx) => <RunTab listing={ctx.listing} setLogFile={ctx.setLogFile} />,
  },
  {
    id: "tools",
    label: "Tools",
    render: (ctx) => <ToolsTab listing={ctx.listing} setLogFile={ctx.setLogFile} />,
  },
];

export default function App() {
  const [listing, setListing] = useState<ProjectListing | null>(null);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [logFile, setLogFile] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>(TABS[0].id);
  // The config file the Config tab edits. null -> fall back to the project's
  // first config; a sidebar click on a config file sets it explicitly.
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);

  const openProject = async (path: string) => {
    try {
      setProjectError(null);
      const result = await api.openProject(path);
      setListing(result);
      setSelectedConfig(null); // let the new project pick its own default config
    } catch (e) {
      setProjectError(String(e instanceof Error ? e.message : e));
      setListing(null);
    }
  };

  // On load, auto-open the project the server was launched with, if any.
  useEffect(() => {
    api.config().then((cfg) => {
      if (cfg.initial_project) openProject(cfg.initial_project);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Clicking a config file opens it in the Config tab (and switches to it);
  // any other file is tailed in the bottom log terminal.
  const onSelectFile = (entry: FileEntry) => {
    if (entry.kind === "config") {
      setSelectedConfig(entry.path);
      setActiveTab("config");
    } else {
      setLogFile(entry.path);
    }
  };

  const projectName = listing ? listing.dir.split("/").pop() || listing.dir : null;
  const current = TABS.find((t) => t.id === activeTab) || TABS[0];
  // The Config tab edits the clicked config, else the project's first config.
  const configPath =
    selectedConfig ??
    (listing && listing.configs.length ? listing.configs[0].path : null);

  return (
    <div className="app">
      <TopBar projectName={projectName} />
      <div className="app-body">
        <Sidebar
          listing={listing}
          onOpen={openProject}
          onSelectFile={onSelectFile}
          error={projectError}
        />
        <main className="workspace">
          <nav className="tabbar">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab ${t.id === activeTab ? "active" : ""}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="tab-content">
            {current.render({ configPath, listing, setLogFile, setActiveTab })}
          </div>
        </main>
      </div>
      <LogTerminal file={logFile} />
    </div>
  );
}
