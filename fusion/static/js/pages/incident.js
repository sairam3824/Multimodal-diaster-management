import { getActiveRecord, getPendingJob, getRecordById, getRecordByJobId, setActiveRecord } from "/static/js/store.js";
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

export function initIncidentPage(){
  const params = new URLSearchParams(window.location.search);
  const pendingJobId = params.get("job");
  const record = resolveRecord();

  if(!record){
    const empty = document.getElementById("incident-empty");
    const pendingJob = pendingJobId ? getPendingJob(pendingJobId) : null;
    empty.style.display = "block";
    if(pendingJob){
      empty.textContent = "Analysis is running in the background. You can keep browsing the platform and this incident page will update automatically when the saved result is ready.";
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
  document.getElementById("incident-briefing").innerHTML = (result.xai?.summary || "No responder briefing available.").replace(/\n/g, "<br>");

  if(result.xai?.gradcam_b64){
    const image = document.getElementById("incident-gradcam");
    image.src = `data:image/png;base64,${result.xai.gradcam_b64}`;
    image.style.display = "block";
  }

  document.getElementById("incident-crisis-category").textContent = titleize(result.crisis?.category || "unknown");
  document.getElementById("incident-crisis-confidence").textContent = formatPercent(result.crisis?.confidence || 0);

  renderProbabilities(result.crisis?.probabilities || {}, result.crisis?.category);

  renderBarRows("incident-modality", [
    {
      label: "Vision",
      width: formatPercent(result.crisis?.vision_weight || 0),
      value: formatPercent(result.crisis?.vision_weight || 0)
    },
    {
      label: "Text",
      width: formatPercent(result.crisis?.text_weight || 0),
      value: formatPercent(result.crisis?.text_weight || 0)
    }
  ]);

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
