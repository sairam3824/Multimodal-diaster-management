const API_BASE = "";

export async function fetchHealth(){
  const response = await fetch(`${API_BASE}/health`);
  if(!response.ok){
    throw new Error("Unable to reach health endpoint");
  }
  return response.json();
}

export async function runAnalysis(formData){
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData
  });

  if(!response.ok){
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Analysis failed");
  }

  return response.json();
}

export async function createAnalysisJob(formData){
  const response = await fetch(`${API_BASE}/analysis/jobs`, {
    method: "POST",
    body: formData
  });

  if(!response.ok){
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Unable to start background analysis");
  }

  return response.json();
}

export async function fetchAnalysisJob(jobId){
  const response = await fetch(`${API_BASE}/analysis/jobs/${jobId}`);
  if(!response.ok){
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Unable to read background analysis job");
  }
  return response.json();
}
