import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// dev server proxies /api to the FastAPI process (python -m hospital_rl.api)
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
