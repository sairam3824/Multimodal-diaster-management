import { createAnalysisJob } from "/static/js/api.js";
import { addPendingJob } from "/static/js/store.js";
import { formatPercent, titleize } from "/static/js/utils.js";

const FIELD_MAP = {
  lat: "s-lat",
  lon: "s-lon",
  precipitation: "s-precip",
  max_temp: "s-maxtemp",
  min_temp: "s-mintemp",
  avg_wind_speed: "s-wind",
  month: "s-month",
  wind_kts: "s-windkts",
  pressure: "s-pressure",
  depth: "s-depth",
  elevation_m: "s-elev",
  distance_to_river_m: "s-river",
  rainfall_7d: "s-r7d",
  monthly_rainfall: "s-rmon",
  ndvi: "s-ndvi",
  ndwi: "s-ndwi"
};

function renderPreview(file){
  const image = document.getElementById("analysis-preview");
  const hint = document.getElementById("analysis-hint");
  const status = document.getElementById("analysis-file-status");
  const dropzone = document.getElementById("analysis-dropzone");
  image.src = URL.createObjectURL(file);
  image.style.display = "block";
  hint.style.display = "none";
  status.textContent = `${file.name} selected`;
  dropzone.classList.remove("drag-over");
}

function setLoading(isLoading){
  const button = document.getElementById("analysis-submit");
  const label = document.getElementById("analysis-submit-label");
  button.disabled = isLoading;
  label.textContent = isLoading ? "Analyzing case" : "Run unified analysis";
}

function showError(message){
  const el = document.getElementById("analysis-error");
  el.textContent = message;
  el.style.display = "block";
}

function clearError(){
  document.getElementById("analysis-error").style.display = "none";
}

const SENSOR_RANGES = {
  lat:                 { min: -90,   max: 90,    label: "Latitude" },
  lon:                 { min: -180,  max: 180,   label: "Longitude" },
  precipitation:       { min: 0,     max: 1000,  label: "Precipitation" },
  max_temp:            { min: -60,   max: 60,    label: "Max temperature" },
  min_temp:            { min: -80,   max: 55,    label: "Min temperature" },
  avg_wind_speed:      { min: 0,     max: 200,   label: "Wind speed" },
  month:               { min: 1,     max: 12,    label: "Month" },
  wind_kts:            { min: 0,     max: 250,   label: "Wind (knots)" },
  pressure:            { min: 850,   max: 1100,  label: "Pressure (hPa)" },
  depth:               { min: 0,     max: 800,   label: "Depth (km)" },
  elevation_m:         { min: -500,  max: 9000,  label: "Elevation (m)" },
  distance_to_river_m: { min: 0,     max: 100000,label: "Distance to river (m)" },
  rainfall_7d:         { min: 0,     max: 2000,  label: "Rainfall 7-day (mm)" },
  monthly_rainfall:    { min: 0,     max: 3000,  label: "Monthly rainfall (mm)" },
  ndvi:                { min: -1,    max: 1,     label: "NDVI" },
  ndwi:                { min: -1,    max: 1,     label: "NDWI" },
};

function collectSensorInput(){
  const sensors = {};
  const errors = [];
  Object.entries(FIELD_MAP).forEach(([key, id]) => {
    const raw = document.getElementById(id).value;
    if(raw !== ""){
      const num = parseFloat(raw);
      if(isNaN(num)){
        errors.push(`${SENSOR_RANGES[key]?.label || key}: must be a number`);
        return;
      }
      const range = SENSOR_RANGES[key];
      if(range && (num < range.min || num > range.max)){
        errors.push(`${range.label}: must be between ${range.min} and ${range.max}`);
        return;
      }
      sensors[key] = num;
    }
  });
  if(errors.length > 0){
    throw new Error("Sensor validation errors:\n" + errors.join("\n"));
  }
  return sensors;
}

function getDroppedImage(dataTransfer){
  if(!dataTransfer) return null;

  if(dataTransfer.items){
    for(const item of dataTransfer.items){
      if(item.kind === "file"){
        const file = item.getAsFile();
        if(file && file.type.startsWith("image/")){
          return file;
        }
      }
    }
  }

  if(dataTransfer.files){
    for(const file of dataTransfer.files){
      if(file.type.startsWith("image/")){
        return file;
      }
    }
  }

  return null;
}

function renderResult(record){
  document.getElementById("analysis-result").style.display = "block";
  document.getElementById("analysis-result-title").textContent = `${record.result.alert_level} alert`;
  document.getElementById("analysis-result-summary").textContent = record.result.summary || "Analysis completed.";
  document.getElementById("analysis-result-type").textContent = titleize(record.result.disaster_type || "unknown");
  document.getElementById("analysis-result-priority").textContent = titleize(record.result.priority || "low");
  document.getElementById("analysis-result-severity").textContent = formatPercent(record.result.fused_severity || 0);
  document.getElementById("analysis-detail-link").href = `/incident?id=${record.id}`;
  document.getElementById("analysis-iot-link").href = `/iot-monitor?id=${record.id}`;
  document.getElementById("analysis-reports-link").href = "/reports";
}

export function initAnalysisPage(){
  const fileInput = document.getElementById("analysis-file");
  const dropzone = document.getElementById("analysis-dropzone");
  const tweetInput = document.getElementById("tweet-input");
  const charCount = document.getElementById("analysis-char-count");
  let selectedFile = null;
  let dragDepth = 0;

  function setSelectedFile(file){
    if(!file || !file.type.startsWith("image/")){
      showError("Please choose an image file in JPG, PNG, or WEBP format.");
      return;
    }

    selectedFile = file;
    renderPreview(selectedFile);
    clearError();

    // Keep the hidden input in sync so browser form semantics remain correct.
    try{
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
    }catch{
      // Some browsers restrict programmatic FileList assignment. Preview/state still work.
    }
  }

  ["dragenter", "dragover", "dragleave", "drop"].forEach(type => {
    document.addEventListener(type, event => {
      event.preventDefault();
    });
  });

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", event => {
    if(event.key === "Enter" || event.key === " "){
      event.preventDefault();
      fileInput.click();
    }
  });
  dropzone.addEventListener("dragenter", event => {
    event.preventDefault();
    dragDepth += 1;
    dropzone.classList.add("drag-over");
  });
  dropzone.addEventListener("dragover", event => {
    event.preventDefault();
    if(event.dataTransfer){
      event.dataTransfer.dropEffect = "copy";
    }
    dropzone.classList.add("drag-over");
  });
  dropzone.addEventListener("dragleave", event => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if(dragDepth === 0){
      dropzone.classList.remove("drag-over");
    }
  });
  dropzone.addEventListener("drop", event => {
    event.preventDefault();
    event.stopPropagation();
    dragDepth = 0;
    dropzone.classList.remove("drag-over");
    const file = getDroppedImage(event.dataTransfer);
    if(file){
      setSelectedFile(file);
    }else{
      showError("Please drop an image file in JPG, PNG, or WEBP format.");
    }
  });

  fileInput.addEventListener("change", () => {
    if(fileInput.files[0]){
      setSelectedFile(fileInput.files[0]);
    }
  });

  tweetInput.addEventListener("input", () => {
    charCount.textContent = `${tweetInput.value.length} / 560`;
  });

  document.getElementById("analysis-form").addEventListener("submit", async event => {
    event.preventDefault();
    clearError();

    if(!selectedFile){
      showError("Add an image before running analysis.");
      return;
    }

    if(!tweetInput.value.trim()){
      showError("Add the public report text before running analysis.");
      return;
    }

    let sensors;
    try{
      sensors = collectSensorInput();
    }catch(validationError){
      showError(validationError.message);
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("tweet", tweetInput.value.trim());
    Object.entries(sensors).forEach(([key, value]) => formData.append(key, value));

    try{
      setLoading(true);
      document.getElementById("analysis-submit-label").textContent = "Scheduling background analysis";
      const job = await createAnalysisJob(formData);
      const pendingJob = addPendingJob({
        jobId: job.job_id,
        createdAt: job.created_at,
        updatedAt: job.updated_at,
        status: job.status,
        input: {
          fileName: selectedFile.name,
          tweet: tweetInput.value.trim(),
          sensors
        }
      });

      renderResult({
        id: pendingJob.jobId,
        result: {
          alert_level: "Queued",
          summary: "The analysis is now running in the background. You can move to any page and the result will be saved automatically when it completes.",
          disaster_type: "pending",
          priority: "pending",
          fused_severity: 0
        }
      });
      document.getElementById("analysis-submit-label").textContent = "Opening incident details";
      window.location.href = `/incident?job=${job.job_id}`;
    }catch(error){
      showError(error.message);
    }finally{
      setLoading(false);
    }
  });
}
