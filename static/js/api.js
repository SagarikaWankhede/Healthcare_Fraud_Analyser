import { buildQuery } from "./format.js";

export class ApiError extends Error {
  constructor(status, message, details = null) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

async function request(method, path, { params, body, timeoutMs = 20000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(`/api${path}${buildQuery(params)}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (err) {
    throw new ApiError(0, err.name === "AbortError" ? "The server took too long to respond." : "Cannot reach the MediGuard API.");
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    throw new ApiError(res.status, (data && data.error) || `Request failed (${res.status})`, data && data.details);
  }
  return data;
}

export const api = {
  get: (path, params) => request("GET", path, { params }),
  post: (path, body) => request("POST", path, { body }),
  patch: (path, body) => request("PATCH", path, { body }),
};
