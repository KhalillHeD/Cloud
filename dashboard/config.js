// Overwritten at deploy time by `python tasks.py deploy` (API URL of the stack).
// To use the dashboard from disk, set apiBase to the HttpApiUrl stack output.
window.APP_CONFIG = window.APP_CONFIG || { apiBase: "", mode: "aws", tickSeconds: 300 };
