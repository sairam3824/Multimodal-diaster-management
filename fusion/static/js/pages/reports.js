import { getHistory, setActiveRecord } from "/static/js/store.js";
import { downloadJson, formatDateTime, formatPercent, titleize } from "/static/js/utils.js";

export function initReportsPage(){
  const renderPage = () => {
    const history = getHistory();
    document.getElementById("reports-count").textContent = `${history.length}`;

    const container = document.getElementById("reports-list");
    if(!history.length){
      container.innerHTML = `<div class="empty-state">No reports are available yet. Each run from New Analysis will be saved here automatically.</div>`;
      return;
    }

    container.innerHTML = `
      <div class="surface pad">
        <table class="table">
          <thead>
            <tr>
              <th>Created</th>
              <th>Alert</th>
              <th>Hazard</th>
              <th>Priority</th>
              <th>Severity</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${history.map(record => `
              <tr>
                <td>${formatDateTime(record.createdAt)}</td>
                <td>${record.result.alert_level}</td>
                <td>${titleize(record.result.disaster_type || "unknown")}</td>
                <td>${titleize(record.result.priority || "low")}</td>
                <td>${formatPercent(record.result.fused_severity || 0)}</td>
                <td><a class="link-text" href="/incident?id=${record.id}" data-id="${record.id}">Open</a></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;

    container.querySelectorAll("[data-id]").forEach(link => {
      link.addEventListener("click", () => setActiveRecord(link.dataset.id));
    });
  };

  document.getElementById("reports-export").addEventListener("click", () => {
    downloadJson("fusion-analysis-history.json", getHistory());
  });

  renderPage();
  window.addEventListener("fusion:job-saved", renderPage);
}
