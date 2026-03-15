import { getActiveRecord, getRecordById, setActiveRecord } from "/static/js/store.js";
import { formatDateTime, formatPercent, titleize } from "/static/js/utils.js";

function resolveRecord(){
  const id = new URLSearchParams(window.location.search).get("id");
  const record = id ? getRecordById(id) : getActiveRecord();
  if(record){
    setActiveRecord(record.id);
  }
  return record;
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

export function initIotMonitorPage(){
  const record = resolveRecord();
  if(!record){
    document.getElementById("iot-empty").style.display = "block";
    document.getElementById("iot-content").style.display = "none";
    return;
  }

  const iot = record.result.iot || {};
  document.getElementById("iot-empty").style.display = "none";
  document.getElementById("iot-content").style.display = "flex";

  document.getElementById("iot-case-type").textContent = titleize(record.result.disaster_type || "unknown");
  document.getElementById("iot-case-time").textContent = formatDateTime(record.createdAt);
  document.getElementById("iot-fire").textContent = formatPercent(iot.fire_prob || 0);
  document.getElementById("iot-storm").textContent = `Cat ${((iot.storm_cat || 0) * 5).toFixed(1)}`;
  document.getElementById("iot-quake").textContent = `M${((iot.eq_magnitude || 0) * 9).toFixed(2)}`;
  document.getElementById("iot-flood").textContent = `${((iot.flood_risk || 0) * 100).toFixed(0)}/100`;
  document.getElementById("iot-casualty").textContent = formatPercent(iot.casualty_risk || 0);

  renderBarRows("iot-weights", [
    { label: "Weather", width: formatPercent(iot.sensor_weights?.weather || 0), value: formatPercent(iot.sensor_weights?.weather || 0) },
    { label: "Storm", width: formatPercent(iot.sensor_weights?.storm || 0), value: formatPercent(iot.sensor_weights?.storm || 0) },
    { label: "Seismic", width: formatPercent(iot.sensor_weights?.seismic || 0), value: formatPercent(iot.sensor_weights?.seismic || 0) },
    { label: "Hydro", width: formatPercent(iot.sensor_weights?.hydro || 0), value: formatPercent(iot.sensor_weights?.hydro || 0) }
  ]);
}
