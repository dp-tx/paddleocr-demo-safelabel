# Label Lens

Label Lens is a static, local-first OCR demo for food packaging, shampoo
bottles, ingredient lists, and similar scene text. It offers two engines:

- **Vision OCR** uses Microsoft Florence-2 through Transformers.js for the best
  reading accuracy. It uses WebGPU and language context to handle difficult
  label text.
- **Classic OCR** uses PaddleOCR.js as a lighter compatibility fallback.

Both engines run inside the browser, so the selected image is never uploaded to
an application server.

## Run locally

You need Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Open the URL printed by Vite, choose a JPG, PNG, or WebP image, and select
an OCR mode, then run the scan. The first Vision OCR scan downloads roughly
340 MB of model files on most GPUs; the browser caches them for later visits.

## Validate a production build

```bash
npm run check
npm run lint
npm run build
npm run preview
```

The production-ready static site is written to `dist/`. Asset URLs are relative,
so that directory works under both root domains and GitHub Pages project paths
such as `https://username.github.io/repository/`.

## Deploy to GitHub Pages

This repository includes `.github/workflows/deploy-pages.yml`.

1. Create a GitHub repository and push this project to its `main` branch.
2. In the repository, open **Settings → Pages**.
3. Set **Source** to **GitHub Actions**.
4. Open the **Actions** tab and run **Deploy to GitHub Pages**, or push another
   commit to `main`.

The workflow installs exact dependencies with `npm ci`, checks TypeScript and
lint rules, builds the static site, and publishes `dist/` using GitHub's official
Pages actions.

## Implementation notes

- `app/page.tsx` contains file validation, crop enhancement, result merging,
  lazy model initialization, clipboard support, and text download.
- `app/florence.worker.ts` runs Florence-2 off the main thread with the
  quantization mix recommended by Hugging Face for WebGPU.
- `src/main.tsx` mounts the React application into the static `index.html` page.
- `vite.config.ts` uses relative asset paths, which is the important setting for
  GitHub Pages subdirectory deployments.
- Both OCR engines run in Web Workers so model work does not freeze the interface.
- Vision OCR requires WebGPU. The UI automatically falls back to Classic OCR
  when WebGPU is unavailable.
- The demo uses the English PP-OCRv6 model. Change `lang` in `getEngine()` for a
  different supported language model.
- The ONNX Runtime WebAssembly binary is loaded from its pinned jsDelivr package
  path. Images remain local even though runtime and model files come from CDNs.

For better label results, fill the camera frame, keep small print sharp, avoid
specular glare, and photograph curved containers from more than one angle.
