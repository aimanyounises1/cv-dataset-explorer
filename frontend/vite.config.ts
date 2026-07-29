import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// API + media requests are proxied to the FastAPI backend in dev.
// Vite exposes existing process environment variables while evaluating this
// config. An override lets a verifier run an isolated backend without stopping
// another developer's server; normal `npm run dev` keeps the documented 8000
// default.
const apiTarget = process.env.CVDE_DEV_API ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": apiTarget,
      "/media": apiTarget,
    },
  },
});
