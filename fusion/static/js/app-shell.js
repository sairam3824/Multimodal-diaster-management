import { fetchAnalysisJob, fetchHealth } from "/static/js/api.js";
import { getPendingJobs, removePendingJob, saveAnalysisRecord, updatePendingJob } from "/static/js/store.js";

const NAV_ITEMS = [
  { key: "overview", label: "Overview", href: "/overview" },
  { key: "analysis", label: "New Analysis", href: "/analysis" },
  { key: "incident", label: "Incident Details", href: "/incident" },
  { key: "iot-monitor", label: "IoT Monitor", href: "/iot-monitor" },
  { key: "reports", label: "Reports", href: "/reports" }
];

function renderHeader(pageKey){
  const header = document.getElementById("app-header");
  if(!header) return;

  const nav = NAV_ITEMS.map(item => `
    <a href="${item.href}" class="${item.key === pageKey ? "active" : ""}">${item.label}</a>
  `).join("");

  header.innerHTML = `
    <header class="site-header">
      <div class="brand">
        <h1>Multimodal Disaster Intelligence Platform</h1>
        <p>IoT sensor fusion, social media analysis, and real-time operational assessment.</p>
      </div>
      <div class="header-spacer"></div>
      <nav class="main-nav">${nav}</nav>
    </header>
  `;
}

function renderFooter(){
  const footer = document.getElementById("app-footer");
  if(!footer) return;

  footer.innerHTML = `
    <footer class="footer">
      <div>2026 Multimodal Disaster Intelligence Platform</div>
      <div class="footer-note">Operational workspace for overview, analysis, incident review, IoT monitoring, and reporting.</div>
    </footer>
  `;
}

async function refreshHealth(){
  const pill = document.getElementById("pipeline-status");
  if(!pill) return;

  try{
    const data = await fetchHealth();
    const pendingCount = getPendingJobs().length;
    if(data.pipeline){
      pill.textContent = pendingCount ? `Pipeline ready · ${pendingCount} running` : "Pipeline ready";
    }else{
      pill.textContent = "Loading pipeline";
    }
    pill.className = `status-pill${data.pipeline ? " online" : ""}`;
  }catch{
    pill.textContent = "Server offline";
    pill.className = "status-pill";
  }
}

async function syncPendingJobs(){
  const pendingJobs = getPendingJobs();
  if(!pendingJobs.length){
    return;
  }

  await Promise.all(pendingJobs.map(async job => {
    try{
      const status = await fetchAnalysisJob(job.jobId);
      if(status.status === "completed" && status.result){
        const record = saveAnalysisRecord(job.input, status.result, { jobId: job.jobId });
        removePendingJob(job.jobId);
        window.dispatchEvent(new CustomEvent("fusion:job-saved", {
          detail: { jobId: job.jobId, record }
        }));
      }else if(status.status === "failed"){
        removePendingJob(job.jobId);
        window.dispatchEvent(new CustomEvent("fusion:job-failed", {
          detail: { jobId: job.jobId, error: status.error || "Background analysis failed" }
        }));
      }else{
        updatePendingJob(job.jobId, {
          status: status.status,
          updatedAt: status.updated_at || new Date().toISOString()
        });
      }
    }catch{
      // Keep the pending job and try again on the next sync cycle.
    }
  }));
}

export async function initShell(pageKey){
  renderHeader(pageKey);
  renderFooter();
  await syncPendingJobs();
  await refreshHealth();
  window.setInterval(async () => {
    await syncPendingJobs();
    await refreshHealth();
  }, 3000);
}
