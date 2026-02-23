import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";
import { existsSync } from "fs";
import runtimeErrorOverlay from "@replit/vite-plugin-runtime-error-modal";
import yaml from '@rollup/plugin-yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Determine shared directory path (Docker vs local)
const dockerSharedPath = path.resolve(__dirname, "project-shared");
const localSharedPath = path.resolve(__dirname, "..", "shared");
const sharedPath = existsSync(dockerSharedPath) ? dockerSharedPath : localSharedPath;

export default defineConfig({
  plugins: [
    react(),
    runtimeErrorOverlay(),
    yaml(), // Enable YAML imports
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "client", "src"),
      "@shared": sharedPath, // Automatically resolves Docker vs local path
      "@assets": path.resolve(__dirname, "attached_assets"),
    },
  },
  root: path.resolve(__dirname, "client"),
  build: {
    outDir: path.resolve(__dirname, "dist/public"),
    emptyOutDir: true,
  },
  server: {
    fs: {
      strict: false,
      allow: [path.resolve(__dirname, "..")], // Allow access to shared directory
    },
  },
});
