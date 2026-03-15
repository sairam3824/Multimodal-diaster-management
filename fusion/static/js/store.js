const HISTORY_KEY = "fusion.analysis.history.v1";
const ACTIVE_KEY = "fusion.analysis.active.v1";
const PENDING_KEY = "fusion.analysis.pending.v1";

function loadHistory(){
  try{
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  }catch{
    return [];
  }
}

function persistHistory(history){
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function loadPending(){
  try{
    const raw = localStorage.getItem(PENDING_KEY);
    return raw ? JSON.parse(raw) : [];
  }catch{
    return [];
  }
}

function persistPending(pending){
  localStorage.setItem(PENDING_KEY, JSON.stringify(pending));
}

export function getHistory(){
  return loadHistory();
}

export function getRecordByJobId(jobId){
  return loadHistory().find(item => item.sourceJobId === jobId) || null;
}

export function saveAnalysisRecord(input, result, meta = {}){
  if(meta.jobId){
    const existing = getRecordByJobId(meta.jobId);
    if(existing){
      localStorage.setItem(ACTIVE_KEY, existing.id);
      return existing;
    }
  }

  const history = loadHistory();
  const record = {
    id: `${Date.now()}`,
    createdAt: new Date().toISOString(),
    input,
    result,
    sourceJobId: meta.jobId || null
  };

  history.unshift(record);
  persistHistory(history.slice(0, 40));
  localStorage.setItem(ACTIVE_KEY, record.id);
  return record;
}

export function getLatestRecord(){
  return loadHistory()[0] || null;
}

export function getRecordById(id){
  return loadHistory().find(item => item.id === id) || null;
}

export function setActiveRecord(id){
  localStorage.setItem(ACTIVE_KEY, id);
}

export function getActiveRecord(){
  const id = localStorage.getItem(ACTIVE_KEY);
  return id ? getRecordById(id) : getLatestRecord();
}

export function getPendingJobs(){
  return loadPending();
}

export function getPendingJob(jobId){
  return loadPending().find(item => item.jobId === jobId) || null;
}

export function addPendingJob(job){
  const pending = loadPending().filter(item => item.jobId !== job.jobId);
  pending.unshift(job);
  persistPending(pending);
  return job;
}

export function updatePendingJob(jobId, updates){
  const pending = loadPending();
  const next = pending.map(item => item.jobId === jobId ? { ...item, ...updates } : item);
  persistPending(next);
  return next.find(item => item.jobId === jobId) || null;
}

export function removePendingJob(jobId){
  const next = loadPending().filter(item => item.jobId !== jobId);
  persistPending(next);
}
