export function titleize(value){
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, char => char.toUpperCase());
}

export function formatPercent(value, digits = 1){
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

export function formatNumber(value, digits = 1){
  return Number(value || 0).toFixed(digits);
}

export function formatDateTime(value){
  if(!value){
    return "No recent activity";
  }

  try{
    return new Intl.DateTimeFormat("en-US", {
      dateStyle: "medium",
      timeStyle: "short"
    }).format(new Date(value));
  }catch{
    return new Date(value).toLocaleString();
  }
}

export function priorityTone(priority){
  const key = String(priority || "").toLowerCase();
  if(key === "critical") return "red";
  if(key === "high") return "gold";
  if(key === "medium") return "gold";
  return "green";
}

export function alertTone(level){
  const key = String(level || "").toLowerCase();
  if(key === "red") return "red";
  if(key === "orange" || key === "yellow") return "gold";
  return "green";
}

export function downloadJson(filename, payload){
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
