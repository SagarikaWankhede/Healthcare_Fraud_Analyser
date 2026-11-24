import { api } from "./api.js";
import { destroyCharts } from "./charts.js";
import { parseRoute } from "./format.js";
import { clear } from "./ui.js";
import * as overview from "./views/overview.js";
import * as audit from "./views/audit.js";
import * as providers from "./views/provider.js";
import * as performance from "./views/performance.js";
import * as methodology from "./views/methodology.js";

const ROUTES = {
  overview: { view: overview, title: "Overview" },
  audit: { view: audit, title: "Audit Priorities" },
  providers: { view: providers, title: "Provider Analysis" },
  performance: { view: performance, title: "Model Performance" },
  methodology: { view: methodology, title: "Methodology" },
};

const root = document.getElementById("view");
let renderToken = 0;

async function render() {
  const route = parseRoute(window.location.hash);
  const entry = ROUTES[route.name] || ROUTES.overview;
  const name = ROUTES[route.name] ? route.name : "overview";
  destroyCharts();
  clear(root);
  document.title = `${entry.title} · MediGuard`;
  for (const a of document.querySelectorAll("[data-nav]")) {
    const active = a.dataset.nav === name;
    a.classList.toggle("active", active);
    if (active) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  const token = ++renderToken;
  // Views receive a guard so late responses from a previous route never paint over the current one.
  const ctx = { param: route.param, isCurrent: () => token === renderToken };
  await entry.view.render(root, ctx);
  window.scrollTo(0, 0);
}

async function loadModelChip() {
  const chip = document.getElementById("model-chip");
  try {
    const h = await api.get("/health");
    chip.textContent = `Model ${h.model_name} · ${h.model_version}`;
  } catch (err) {
    chip.textContent = "Model artifacts unavailable";
  }
}

window.addEventListener("hashchange", render);
if (!window.location.hash) window.history.replaceState(null, "", "#/overview");
render();
loadModelChip();
