import { getHistory, getLatestRecord, setActiveRecord } from "/static/js/store.js";
import { alertTone, formatDateTime, formatPercent, priorityTone, titleize } from "/static/js/utils.js";

function renderCard(id, label, value, note, tone){
  const target = document.getElementById(id);
  if(!target) return;

  target.innerHTML = `
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
    <div class="metric-note">${note}</div>
  `;

  if(tone){
    target.querySelector(".metric-value").classList.add(tone === "red" ? "danger-color" : tone === "gold" ? "fire-color" : "flood-color");
  }
}

function renderRecent(history){
  const container = document.getElementById("recent-list");
  if(!container) return;

  if(!history.length){
    container.innerHTML = `<div class="empty-state">No analyses have been run yet. Start from the New Analysis page to create your first incident record.</div>`;
    return;
  }

  container.innerHTML = history.slice(0, 5).map(record => `
    <div class="list-row">
      <div class="list-copy">
        <div class="list-title">${titleize(record.result.disaster_type || "unknown")} - ${record.result.alert_level} alert</div>
        <div class="list-subtitle">${record.result.summary || "No summary available"}<br>${formatDateTime(record.createdAt)}</div>
      </div>
      <a class="button secondary" href="/incident?id=${record.id}" data-id="${record.id}">Open</a>
    </div>
  `).join("");

  container.querySelectorAll("[data-id]").forEach(link => {
    link.addEventListener("click", () => setActiveRecord(link.dataset.id));
  });
}

export function initOverviewPage(){
  const renderPage = () => {
    const latest = getLatestRecord();
    const history = getHistory();

    renderCard(
      "overview-alert",
      "Current alert",
      latest ? `${latest.result.alert_level} alert` : "No active case",
      latest ? latest.result.summary : "Run a case to generate a live operational alert.",
      latest ? alertTone(latest.result.alert_level) : "green"
    );

    renderCard(
      "overview-type",
      "Detected hazard",
      latest ? titleize(latest.result.disaster_type || "unknown") : "Awaiting analysis",
      latest ? `Priority ${titleize(latest.result.priority || "low")}` : "Hazard type appears after fusion analysis."
    );

    renderCard(
      "overview-priority",
      "Response priority",
      latest ? titleize(latest.result.priority || "low") : "No priority",
      latest ? `${formatPercent(latest.result.fused_severity)} fused severity` : "Priority is set when a case is analyzed.",
      latest ? priorityTone(latest.result.priority) : "green"
    );

    renderCard(
      "overview-updated",
      "Latest update",
      latest ? formatDateTime(latest.createdAt) : "No updates yet",
      `${history.length} saved incident records are available in reports.`
    );

    renderRecent(history);
  };

  renderPage();
  window.addEventListener("fusion:job-saved", renderPage);
}
