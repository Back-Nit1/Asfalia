import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Artifacts from "./pages/Artifacts";
import Build from "./pages/Build";
import Experiments from "./pages/Experiments";
import Infrastructure from "./pages/Infrastructure";
import Overview from "./pages/Overview";
import Training from "./pages/Training";
import World from "./pages/World";

const NAV = [["/overview", "Overview"], ["/build", "Build"], ["/training", "Training"], ["/world/fixture_tiny_episode", "World"],
  ["/experiments", "Experiments"], ["/infrastructure", "Infrastructure"], ["/artifacts", "Artifacts"]] as const;

export default function App() {
  return (
    <div className="shell">
      <a className="skip" href="#main">Skip to content</a>
      <nav aria-label="Primary">
        <div className="brand">hospital-rl<span>console</span></div>
        {NAV.map(([to, label]) => (
          <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>{label}</NavLink>
        ))}
      </nav>
      <main id="main">
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/build" element={<Build />} />
          <Route path="/training" element={<Training />} />
          <Route path="/world/:episodeId" element={<World />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/infrastructure" element={<Infrastructure />} />
          <Route path="/artifacts" element={<Artifacts />} />
          <Route path="*" element={<p>Unknown route.</p>} />
        </Routes>
      </main>
    </div>
  );
}
