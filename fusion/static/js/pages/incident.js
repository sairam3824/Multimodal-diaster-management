import { fetchAnalysisJob } from "/static/js/api.js";
import { getActiveRecord, getPendingJob, getRecordById, getRecordByJobId, saveAnalysisRecord, setActiveRecord } from "/static/js/store.js";
import { formatDateTime, formatPercent, titleize } from "/static/js/utils.js";

function alertClass(level){
  const key = String(level || "").toLowerCase();
  if(key === "green") return "alert-green";
  if(key === "yellow") return "alert-yellow";
  if(key === "orange") return "alert-orange";
  if(key === "red") return "alert-red";
  return "";
}

function resolveRecord(){
  const params = new URLSearchParams(window.location.search);
  const id = params.get("id");
  const jobId = params.get("job");
  const record = id ? getRecordById(id) : null;
  const jobRecord = !id && jobId ? getRecordByJobId(jobId) : null;
  const resolved = id ? record : (jobId ? jobRecord : getActiveRecord());
  if(resolved){
    setActiveRecord(resolved.id);
  }
  return resolved;
}

function renderProbabilities(probabilities, category){
  const container = document.getElementById("incident-probabilities");
  const entries = Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]);
  container.innerHTML = entries.map(([label, value]) => `
    <div class="probability-row${label === category ? " top" : ""}">
      <div>${titleize(label)}</div>
      <div class="bar"><div class="bar-fill" style="width:${(value * 100).toFixed(1)}%"></div></div>
      <div>${formatPercent(value)}</div>
    </div>
  `).join("");
}

function damageSlug(label){
  return label.replace(/[\s_]+/g, "-").toLowerCase();
}

function renderSatelliteProbabilities(damageProbs, predictedClass){
  const section = document.getElementById("incident-sat-probabilities-section");
  section.style.display = "flex";

  // Predicted class badge
  const badge = document.getElementById("incident-sat-predicted");
  badge.textContent = titleize(predictedClass);
  badge.className = "metric-value damage-badge damage-" + damageSlug(predictedClass);

  // Probability bars sorted by value
  const container = document.getElementById("incident-sat-probability-list");
  const entries = Object.entries(damageProbs || {}).sort((a, b) => b[1] - a[1]);
  container.innerHTML = entries.map(([label, value]) => {
    const slug = damageSlug(label);
    const isTop = damageSlug(label) === damageSlug(predictedClass);
    return `
      <div class="probability-row damage-${slug}${isTop ? " top" : ""}">
        <div>${titleize(label)}</div>
        <div class="bar"><div class="bar-fill" style="width:${(value * 100).toFixed(1)}%"></div></div>
        <div>${formatPercent(value)}</div>
      </div>
    `;
  }).join("");
}

function renderBarRows(containerId, rows){
  const container = document.getElementById(containerId);
  container.innerHTML = rows.map(row => `
    <div class="split-bar-row">
      <div class="label">${row.label}</div>
      <div class="bar"><div class="bar-fill" style="width:${row.width}"></div></div>
      <div class="value">${row.value}</div>
    </div>
  `).join("");
}

function escapeHtml(value){
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatBriefingText(value){
  const escaped = escapeHtml(value || "No responder briefing available.");
  return escaped
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

function briefingStatusLabel(status, summary){
  if(summary) return "Ready";
  if(status === "pending" || status === "running") return "Generating";
  if(status === "skipped") return "Skipped";
  return "Unavailable";
}

function renderBriefing(result){
  const xai = result.xai || {};
  const summary = xai.summary || "";
  const status = xai.summary_status || (summary ? "completed" : "skipped");
  const statusEl = document.getElementById("incident-briefing-status");
  const briefingEl = document.getElementById("incident-briefing");

  if(statusEl){
    statusEl.textContent = briefingStatusLabel(status, summary);
    statusEl.dataset.status = summary ? "completed" : status;
  }

  if(briefingEl){
    briefingEl.innerHTML = summary
      ? formatBriefingText(summary)
      : formatBriefingText(status === "pending" || status === "running"
        ? "Responder briefing is generating in the background."
        : "No responder briefing available.");
  }
}

function initBriefingToggle(){
  const toggle = document.getElementById("incident-briefing-toggle");
  const panel = document.getElementById("incident-briefing-panel");
  if(!toggle || !panel) return;

  toggle.addEventListener("click", () => {
    const isOpen = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!isOpen));
    panel.hidden = isOpen;
  });
}

function startBriefingPolling(record){
  if(!record.sourceJobId) return;

  const initialStatus = record.result?.xai?.summary_status;
  if(record.result?.xai?.summary || !["pending", "running"].includes(initialStatus)){
    return;
  }

  let attempts = 0;
  const timer = window.setInterval(async () => {
    attempts += 1;
    try{
      const job = await fetchAnalysisJob(record.sourceJobId);
      if(job.result){
        renderBriefing(job.result);
        saveAnalysisRecord(record.input, job.result, { jobId: record.sourceJobId });
        const nextStatus = job.result.xai?.summary_status;
        if(job.result.xai?.summary || !["pending", "running"].includes(nextStatus)){
          window.clearInterval(timer);
        }
      }
    }catch(err){
      console.warn(`[incident] Briefing poll failed: ${err.message}`);
    }

    if(attempts >= 40){
      window.clearInterval(timer);
    }
  }, 3000);
}

export function initIncidentPage(){
  initBriefingToggle();

  const params = new URLSearchParams(window.location.search);
  const pendingJobId = params.get("job");
  const record = resolveRecord();

  if(!record){
    const empty = document.getElementById("incident-empty");
    const pendingJob = pendingJobId ? getPendingJob(pendingJobId) : null;
    empty.style.display = "block";
    if(pendingJob){
      empty.textContent = "Incident prediction is running in the background. This page will open the predicted incident as soon as the tri-fusion result is ready.";
      window.addEventListener("fusion:job-saved", event => {
        if(event.detail.jobId === pendingJobId){
          window.location.replace(`/incident?id=${event.detail.record.id}`);
        }
      });
      window.addEventListener("fusion:job-failed", event => {
        if(event.detail.jobId === pendingJobId){
          empty.textContent = `Background analysis failed: ${event.detail.error}`;
        }
      });
    }
    document.getElementById("incident-content").style.display = "none";
    return;
  }

  const result = record.result;
  document.getElementById("incident-empty").style.display = "none";
  document.getElementById("incident-content").style.display = "flex";

  const incidentAlert = document.getElementById("incident-alert");
  incidentAlert.textContent = `${result.alert_level} alert`;
  incidentAlert.className = alertClass(result.alert_level);
  document.getElementById("incident-summary").textContent = result.summary || "No summary returned.";
  document.getElementById("incident-type").textContent = titleize(result.disaster_type || "unknown");
  document.getElementById("incident-priority").textContent = titleize(result.priority || "low");
  document.getElementById("incident-updated").textContent = formatDateTime(record.createdAt);
  document.getElementById("incident-severity").textContent = formatPercent(result.fused_severity || 0);
  renderBriefing(result);
  startBriefingPolling(record);

  // Crisis Grad-CAM
  const crisisGradcam = result.xai?.crisis_gradcam_b64 || result.xai?.gradcam_b64;
  if(crisisGradcam){
    const image = document.getElementById("incident-gradcam");
    image.src = `data:image/png;base64,${crisisGradcam}`;
    image.style.display = "block";
  }

  document.getElementById("incident-crisis-category").textContent = titleize(result.crisis?.category || "unknown");
  document.getElementById("incident-crisis-confidence").textContent = formatPercent(result.crisis?.confidence || 0);

  // Satellite section — top-level card + Grad-CAM
  if(result.satellite && result.satellite.damage_class){
    const satSection = document.getElementById("incident-satellite-section");
    satSection.style.display = "flex";
    document.getElementById("incident-sat-damage").textContent = titleize(result.satellite.damage_class);

    if(result.satellite.damage_probs){
      renderBarRows("incident-sat-probs", Object.entries(result.satellite.damage_probs).map(([key, value]) => ({
        label: titleize(key),
        width: formatPercent(value || 0),
        value: formatPercent(value || 0)
      })));
    }

    // Satellite Grad-CAM
    if(result.xai?.satellite_gradcam_b64){
      const satImage = document.getElementById("incident-sat-gradcam");
      satImage.src = `data:image/png;base64,${result.xai.satellite_gradcam_b64}`;
      satImage.style.display = "block";
    }

    // Full satellite damage probability section (like crisis probabilities)
    if(result.satellite.damage_probs){
      renderSatelliteProbabilities(result.satellite.damage_probs, result.satellite.damage_class);
    }
  }

  // Active modalities
  const modalities = result.active_modalities || ["crisis"];
  document.getElementById("incident-active-modalities").textContent = modalities.map(m => titleize(m)).join(", ");

  renderProbabilities(result.crisis?.probabilities || {}, result.crisis?.category);

  // Fusion gate weights (tri-fusion modality contributions)
  const mw = result.modality_weights || {};
  if(mw.crisis !== undefined){
    renderBarRows("incident-modality", [
      { label: "Crisis", width: formatPercent(mw.crisis || 0), value: formatPercent(mw.crisis || 0) },
      { label: "IoT", width: formatPercent(mw.iot || 0), value: formatPercent(mw.iot || 0) },
      { label: "Satellite", width: formatPercent(mw.satellite || 0), value: formatPercent(mw.satellite || 0) }
    ]);
  }else{
    renderBarRows("incident-modality", [
      { label: "Vision", width: formatPercent(result.crisis?.vision_weight || 0), value: formatPercent(result.crisis?.vision_weight || 0) },
      { label: "Text", width: formatPercent(result.crisis?.text_weight || 0), value: formatPercent(result.crisis?.text_weight || 0) }
    ]);
  }

  renderBarRows("incident-sensors", [
    {
      label: "Weather",
      width: formatPercent(result.iot?.sensor_weights?.weather || 0),
      value: formatPercent(result.iot?.sensor_weights?.weather || 0)
    },
    {
      label: "Storm",
      width: formatPercent(result.iot?.sensor_weights?.storm || 0),
      value: formatPercent(result.iot?.sensor_weights?.storm || 0)
    },
    {
      label: "Seismic",
      width: formatPercent(result.iot?.sensor_weights?.seismic || 0),
      value: formatPercent(result.iot?.sensor_weights?.seismic || 0)
    },
    {
      label: "Hydro",
      width: formatPercent(result.iot?.sensor_weights?.hydro || 0),
      value: formatPercent(result.iot?.sensor_weights?.hydro || 0)
    }
  ]);

  renderBarRows("incident-resources", Object.entries(result.fusion?.resource_needs || {}).map(([key, value]) => ({
    label: titleize(key),
    width: formatPercent(value || 0),
    value: formatPercent(value || 0)
  })));
}
