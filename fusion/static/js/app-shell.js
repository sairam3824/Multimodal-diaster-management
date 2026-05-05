import { fetchAnalysisJob, fetchHealth } from "/static/js/api.js";
import { getPendingJobs, removePendingJob, saveAnalysisRecord, updatePendingJob } from "/static/js/store.js";

const NAV_ITEMS = [
  { key: "overview", label: "Overview", href: "/overview" },
  { key: "analysis", label: "New Analysis", href: "/analysis" },
  { key: "incident", label: "Incident Details", href: "/incident" },
  { key: "iot-monitor", label: "IoT Monitor", href: "/iot-monitor" },
  { key: "reports", label: "Reports", href: "/reports" }
];

const STATIC_PAGE_HREFS = {
  "/overview": "/static/pages/overview.html",
  "/analysis": "/static/pages/analysis.html",
  "/incident": "/static/pages/incident.html",
  "/iot-monitor": "/static/pages/iot-monitor.html",
  "/reports": "/static/pages/reports.html"
};



function isStaticPreview(){
  return window.location.pathname.startsWith("/static/pages/");
}

function resolveHref(href){
  if(!isStaticPreview()) return href;
  const [path, query = ""] = href.split("?");
  const staticPath = STATIC_PAGE_HREFS[path];
  if(!staticPath) return href;
  return query ? `${staticPath}?${query}` : staticPath;
}



function renderHeader(pageKey){
  const header = document.getElementById("app-header");
  if(!header) return;

  const nav = NAV_ITEMS.map(item => `
    <a href="${resolveHref(item.href)}" class="${item.key === pageKey ? "active" : ""}">${item.label}</a>
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
    <footer class="site-footer">
      <div class="site-footer__divider"></div>
      <div class="site-footer__body">

        <div class="site-footer__brand">
          <div class="site-footer__brand-name">Multimodal Disaster Intelligence Platform</div>
          <p class="site-footer__brand-desc">
            A tri-modal deep learning system fusing social media imagery, IoT sensor streams,
            and satellite data for real-time disaster assessment and responder briefing.
          </p>
        </div>

        <div class="site-footer__nav-group">
          <div class="site-footer__nav-label">Platform</div>
          <a href="${resolveHref("/overview")}" class="site-footer__nav-link">Overview</a>
          <a href="${resolveHref("/analysis")}" class="site-footer__nav-link">New Analysis</a>
          <a href="${resolveHref("/incident")}" class="site-footer__nav-link">Incident Details</a>
          <a href="${resolveHref("/iot-monitor")}" class="site-footer__nav-link">IoT Monitor</a>
          <a href="${resolveHref("/reports")}" class="site-footer__nav-link">Reports</a>
        </div>



      </div>

      <div class="site-footer__bottom">
        <div class="site-footer__copy">&copy; 2026 Multimodal Disaster Intelligence Platform</div>
        <div class="site-footer__sdp">Developed as an SDP Project</div>
      </div>
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
    }catch(err){
      console.warn(`[sync] Job ${job.jobId} fetch failed, will retry:`, err.message);
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
