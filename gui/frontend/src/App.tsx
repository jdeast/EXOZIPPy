import { useEffect, useState } from "react";
import TopBar from "./components/TopBar";
import Sidebar from "./components/Sidebar";
import LogTerminal from "./components/LogTerminal";
import ConfigTab from "./components/ConfigTab";
import TuneTab from "./components/TuneTab";
import ToolsTab from "./components/ToolsTab";
import RunControl from "./components/RunControl";
import { api, type ProjectListing, type FileEntry } from "./api";

// Application shell: top bar + left sidebar + center tabbed workspace + bottom
// log terminal. Landing: Tune when the app knows which config to run (explicit
// file on the command line, or a project with exactly one config) -- the Tune
// tab auto-Solves, so the first thing on screen is the data with the model
// over it. Otherwise Config, whose empty state doubles as the welcome screen.

// Shared context each tab's render receives, so a tab can read the open project,
// attach the bottom terminal to a log file it cares about, and (Config tab) edit
// the project's config file.
interface TabContext {
  listing: ProjectListing | null;
  setLogFile: (file: string | null) => void;
  configPath: string | null;
  setActiveTab: (id: string) => void;
  active: boolean;
}

interface Tab {
  id: string;
  label: string;
  render: (ctx: TabContext) => JSX.Element;
}

const TABS: Tab[] = [
  {
    id: "config",
    label: "Config",
    render: (ctx) => <ConfigTab configPath={ctx.configPath} active={ctx.active} />,
  },
  { id: "tune", label: "Tune", render: (ctx) => <TuneTab configPath={ctx.configPath} /> },
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
  // Tabs stay mounted once visited (hidden via CSS, not unmounted) so their
  // state -- config selection, tune plots, scroll -- survives tab switches.
  const [visited, setVisited] = useState<Set<string>>(new Set([TABS[0].id]));
  // The config file the Config tab edits. null -> fall back to the project's
  // first config; a sidebar click on a config file sets it explicitly.
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);

  const activateTab = (id: string) => {
    setVisited((prev) => (prev.has(id) ? prev : new Set(prev).add(id)));
    setActiveTab(id);
  };

  // Plotly sizes charts at render time; a chart drawn (or resized) while its
  // tab was hidden has a stale size, so poke a resize when the tab reappears.
  useEffect(() => {
    const t = setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
    return () => clearTimeout(t);
  }, [activeTab]);

  const openProject = async (path: string) => {
    try {
      setProjectError(null);
      const result = await api.openProject(path);
      setListing(result);
      setSelectedConfig(null); // let the new project pick its own default config
      return result;
    } catch (e) {
      setProjectError(String(e instanceof Error ? e.message : e));
      setListing(null);
      return null;
    }
  };

  // On load, auto-open the project the server was launched with, if any.
  // Landing: when the app knows which config to run -- an explicit file on the
  // command line (`exozippy-gui kelt4.yaml`) or a project with exactly one
  // config -- go straight to the Tune tab, which auto-Solves it. A project
  // with several configs (or none) lands on Config so the user picks/builds
  // one first.
  useEffect(() => {
    api.config().then(async (cfg) => {
      if (!cfg.initial_project) return;
      const result = await openProject(cfg.initial_project);
      if (cfg.initial_config) {
        setSelectedConfig(cfg.initial_config);
        activateTab("tune");
      } else if (result && result.configs.length === 1) {
        activateTab("tune");
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Clicking a config file opens it in the Config tab (and switches to it);
  // any other file is tailed in the bottom log terminal.
  const onSelectFile = (entry: FileEntry) => {
    if (entry.kind === "config") {
      setSelectedConfig(entry.path);
      activateTab("config");
    } else {
      setLogFile(entry.path);
    }
  };

  const projectName = listing ? listing.dir.split("/").pop() || listing.dir : null;
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
          onOpen={async (path) => {
            // Same landing rule as startup: one config -> Tune (auto-Solve),
            // otherwise Config to pick/build one.
            const result = await openProject(path);
            if (result) activateTab(result.configs.length === 1 ? "tune" : "config");
          }}
          onSelectFile={onSelectFile}
          error={projectError}
        />
        <main className="workspace">
          <nav className="tabbar">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab ${t.id === activeTab ? "active" : ""}`}
                onClick={() => activateTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          {TABS.filter((t) => visited.has(t.id)).map((t) => (
            <div
              key={t.id}
              className="tab-content"
              style={t.id === activeTab ? undefined : { display: "none" }}
            >
              {t.render({
                configPath,
                listing,
                setLogFile,
                setActiveTab: activateTab,
                active: t.id === activeTab,
              })}
            </div>
          ))}
          <RunControl
            configPath={configPath}
            listing={listing}
            setLogFile={setLogFile}
          />
        </main>
      </div>
      <LogTerminal file={logFile} />
    </div>
  );
}
