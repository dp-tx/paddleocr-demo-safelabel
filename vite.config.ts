import { sites } from "@openai/sites-vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths let the same build work at both user and project
  // GitHub Pages URLs, including /owner/repository/ subdirectories.
  base: "./",
  plugins: [
    react(),
    sites(),
    {
      name: "omit-external-onnx-wasm",
      generateBundle(_options, bundle) {
        // PaddleOCR loads this binary from the pinned CDN path in app/page.tsx.
        // Omitting Vite's duplicate keeps the Pages artifact much smaller.
        for (const fileName of Object.keys(bundle)) {
          if (/ort-wasm.*\.wasm$/i.test(fileName)) delete bundle[fileName];
        }
      },
    },
  ],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
