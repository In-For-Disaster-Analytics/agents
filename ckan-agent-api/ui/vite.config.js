import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API calls to the FastAPI service so the browser can use same-origin
// relative paths (/v1/...). Override the target with VITE_API_TARGET if the API runs elsewhere.
const API_TARGET = process.env.VITE_API_TARGET || "http://localhost:8787";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: API_TARGET, changeOrigin: true },
    },
  },
});
