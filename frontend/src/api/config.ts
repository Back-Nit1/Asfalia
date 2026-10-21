// Update rates (Hz). Visuals 2-5 Hz, metrics ~1 Hz; override with ?hz=4&mhz=1 or localStorage keys.
function num(key: string, fallback: number, lo: number, hi: number): number {
  let raw: string | null = null;
  try {
    raw = new URLSearchParams(location.search).get(key) ?? localStorage.getItem(`cfg.${key}`);
  } catch { /* storage unavailable */ }
  const v = raw === null ? NaN : Number(raw);
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
}
export const visualHz = num("hz", 4, 2, 5);
export const metricsHz = num("mhz", 1, 0.2, 2);
