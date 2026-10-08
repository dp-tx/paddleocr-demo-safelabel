import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import type {
  OcrResult,
  OcrResultItem,
  OcrRuntimeParamsInput,
  Point2D,
} from "@paddleocr/paddleocr-js";

type ScanPhase = "idle" | "loading-model" | "recognizing" | "complete" | "error";
type ScanMode = "vision" | "classic";
type ScanResult = OcrResult & { engine: ScanMode };

type FlorencePass = {
  labels: string[];
  quadBoxes: number[][];
  width: number;
  height: number;
};

type FlorenceResponse = {
  passes: FlorencePass[];
  totalMs: number;
};

type FlorenceProgress = {
  status: string;
  message?: string;
  progress?: { progress?: number };
};

// Keeping the engine behind a small interface makes the page easier to read and
// prevents browser-only implementation details from leaking into the UI code.
type OcrEngine = {
  predict: (input: unknown, params?: OcrRuntimeParamsInput) => Promise<OcrResult[]>;
  dispose: () => Promise<void>;
};

type CropRegion = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type ScanSource = {
  input: File | HTMLCanvasElement;
  // PaddleOCR returns coordinates in the source image's coordinate system.
  // This transform maps a cropped/enlarged pass back onto the original photo.
  crop: CropRegion;
  outputWidth: number;
  outputHeight: number;
};

type PositionedItem = OcrResultItem & {
  bounds: ReturnType<typeof getBounds>;
};

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ENHANCED_PASS_LONG_SIDE = 1800;
const WEBGPU_AVAILABLE = typeof navigator !== "undefined" && "gpu" in navigator;

// PaddleOCR's general-purpose defaults resize the longest image side to 960px
// and discard detection boxes below 0.6 confidence. Product labels often have
// faint, tiny print, so label mode deliberately favors recall over speed.
const LABEL_OCR_PARAMS: OcrRuntimeParamsInput = {
  textDetLimitSideLen: 2048,
  textDetLimitType: "max",
  textDetMaxSideLimit: 2560,
  textDetThresh: 0.2,
  textDetBoxThresh: 0.35,
  textDetUnclipRatio: 1.7,
  textRecScoreThresh: 0.12,
};

const phaseCopy: Record<ScanPhase, string> = {
  idle: "Ready",
  "loading-model": "Loading model",
  recognizing: "Reading label",
  complete: "Complete",
  error: "Needs attention",
};

function readableError(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "Something went wrong while reading this image. Please try another photo.";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function getBounds(poly: Point2D[]) {
  const xs = poly.map(([x]) => x);
  const ys = poly.map(([, y]) => y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);

  return {
    left,
    right,
    top,
    bottom,
    width: right - left,
    height: bottom - top,
  };
}

function normalizeText(text: string) {
  return text.toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
}

function editSimilarity(first: string, second: string) {
  if (first === second) return 1;
  if (!first.length || !second.length) return 0;

  // A pair of one-dimensional rows keeps this inexpensive even for long label lines.
  let previous = Array.from({ length: second.length + 1 }, (_, index) => index);
  for (let firstIndex = 1; firstIndex <= first.length; firstIndex += 1) {
    const current = [firstIndex];
    for (let secondIndex = 1; secondIndex <= second.length; secondIndex += 1) {
      const substitutionCost = first[firstIndex - 1] === second[secondIndex - 1] ? 0 : 1;
      current[secondIndex] = Math.min(
        current[secondIndex - 1] + 1,
        previous[secondIndex] + 1,
        previous[secondIndex - 1] + substitutionCost,
      );
    }
    previous = current;
  }

  return 1 - previous[second.length] / Math.max(first.length, second.length);
}

function intersectionOverSmaller(first: PositionedItem, second: PositionedItem) {
  const width = Math.max(
    0,
    Math.min(first.bounds.right, second.bounds.right) - Math.max(first.bounds.left, second.bounds.left),
  );
  const height = Math.max(
    0,
    Math.min(first.bounds.bottom, second.bounds.bottom) - Math.max(first.bounds.top, second.bounds.top),
  );
  const intersection = width * height;
  const firstArea = Math.max(1, first.bounds.width * first.bounds.height);
  const secondArea = Math.max(1, second.bounds.width * second.bounds.height);

  return intersection / Math.min(firstArea, secondArea);
}

function areDuplicateLines(first: PositionedItem, second: PositionedItem) {
  const firstText = normalizeText(first.text);
  const secondText = normalizeText(second.text);
  if (!firstText || !secondText) return false;

  const overlap = intersectionOverSmaller(first, second);
  if (overlap < 0.38) return false;

  const oneContainsTheOther = firstText.includes(secondText) || secondText.includes(firstText);
  return oneContainsTheOther || editSimilarity(firstText, secondText) >= 0.56;
}

function mapItemToOriginal(item: OcrResultItem, source: ScanSource): PositionedItem {
  const scaleX = source.crop.width / source.outputWidth;
  const scaleY = source.crop.height / source.outputHeight;
  const poly = item.poly.map(([x, y]) => [
    source.crop.x + x * scaleX,
    source.crop.y + y * scaleY,
  ] as Point2D);

  return { ...item, poly, bounds: getBounds(poly) };
}

function chooseBetterLine(first: PositionedItem, second: PositionedItem) {
  const firstText = normalizeText(first.text);
  const secondText = normalizeText(second.text);
  const firstValue = first.score + Math.min(firstText.length, 100) * 0.002;
  const secondValue = second.score + Math.min(secondText.length, 100) * 0.002;
  return secondValue > firstValue ? second : first;
}

function sortPositionedItems(items: PositionedItem[]) {
  items.sort((first, second) => {
    const rowTolerance = Math.max(8, Math.min(first.bounds.height, second.bounds.height) * 0.55);
    if (Math.abs(first.bounds.top - second.bounds.top) <= rowTolerance) {
      return first.bounds.left - second.bounds.left;
    }
    return first.bounds.top - second.bounds.top;
  });
}

function addOrReplaceDuplicate(items: PositionedItem[], candidate: PositionedItem) {
  const duplicateIndex = items.findIndex((line) => areDuplicateLines(line, candidate));
  if (duplicateIndex === -1) {
    items.push(candidate);
  } else {
    items[duplicateIndex] = chooseBetterLine(items[duplicateIndex], candidate);
  }
}

function mergeScanResults(results: OcrResult[], sources: ScanSource[], image: CropRegion): ScanResult {
  const merged: PositionedItem[] = [];

  results.forEach((result, resultIndex) => {
    const source = sources[resultIndex];
    if (!source) return;

    result.items.forEach((item) => {
      const candidate = mapItemToOriginal(item, source);
      addOrReplaceDuplicate(merged, candidate);
    });
  });

  sortPositionedItems(merged);

  const firstResult = results[0];
  if (!firstResult) throw new Error("PaddleOCR did not return a result.");

  return {
    engine: "classic",
    image: { width: image.width, height: image.height },
    items: merged.map((item) => ({ poly: item.poly, text: item.text, score: item.score })),
    metrics: {
      detMs: firstResult.metrics.detMs,
      recMs: firstResult.metrics.recMs,
      totalMs: firstResult.metrics.totalMs,
      detectedBoxes: results.reduce((total, result) => total + result.metrics.detectedBoxes, 0),
      recognizedCount: merged.length,
    },
    runtime: firstResult.runtime,
  };
}

function mergeFlorenceResults(
  passes: FlorencePass[],
  sources: ScanSource[],
  image: CropRegion,
  totalMs: number,
): ScanResult {
  const merged: PositionedItem[] = [];

  passes.forEach((pass, passIndex) => {
    const source = sources[passIndex];
    if (!source) return;

    pass.labels.forEach((text, labelIndex) => {
      const coordinates = pass.quadBoxes[labelIndex];
      if (!text.trim() || !coordinates || coordinates.length !== 8) return;

      const poly = [0, 2, 4, 6].map((offset) => [
        coordinates[offset],
        coordinates[offset + 1],
      ] as Point2D);
      const candidate = mapItemToOriginal({ poly, text: text.trim(), score: 1 }, source);
      addOrReplaceDuplicate(merged, candidate);
    });
  });

  sortPositionedItems(merged);

  return {
    engine: "vision",
    image: { width: image.width, height: image.height },
    items: merged.map((item) => ({ poly: item.poly, text: item.text, score: item.score })),
    metrics: {
      detMs: 0,
      recMs: totalMs,
      totalMs,
      detectedBoxes: merged.length,
      recognizedCount: merged.length,
    },
    runtime: {
      requestedBackend: "webgpu",
      detProvider: "Florence-2",
      recProvider: "Florence-2",
      webgpuAvailable: true,
    },
  };
}

function createCropRegions(width: number, height: number): CropRegion[] {
  const ratio = width / height;

  if (ratio < 0.82) {
    const sideInset = Math.round(width * 0.04);
    const cropHeight = Math.round(height * 0.52);
    return [0, 0.24, 0.48].map((position) => ({
      x: sideInset,
      y: Math.round(height * position),
      width: width - sideInset * 2,
      height: cropHeight,
    }));
  }

  if (ratio > 1.22) {
    const verticalInset = Math.round(height * 0.04);
    const cropWidth = Math.round(width * 0.52);
    return [0, 0.24, 0.48].map((position) => ({
      x: Math.round(width * position),
      y: verticalInset,
      width: cropWidth,
      height: height - verticalInset * 2,
    }));
  }

  const insetX = Math.round(width * 0.03);
  const insetY = Math.round(height * 0.03);
  return [{
    x: insetX,
    y: insetY,
    width: width - insetX * 2,
    height: height - insetY * 2,
  }];
}

function locallyNormalizeContrast(context: CanvasRenderingContext2D, width: number, height: number) {
  const image = context.getImageData(0, 0, width, height);
  const grayscale = new Uint8Array(width * height);
  const integral = new Uint32Array((width + 1) * (height + 1));

  for (let y = 0; y < height; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < width; x += 1) {
      const pixelIndex = (y * width + x) * 4;
      const luminance = Math.round(
        image.data[pixelIndex] * 0.299
          + image.data[pixelIndex + 1] * 0.587
          + image.data[pixelIndex + 2] * 0.114,
      );
      grayscale[y * width + x] = luminance;
      rowSum += luminance;
      integral[(y + 1) * (width + 1) + x + 1] = integral[y * (width + 1) + x + 1] + rowSum;
    }
  }

  // Removing slow illumination changes is especially useful on curved bottles:
  // the label becomes nearly white while ink remains dark and well-defined.
  const radius = clamp(Math.round(Math.min(width, height) / 42), 18, 42);
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - radius);
    const bottom = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - radius);
      const right = Math.min(width - 1, x + radius);
      const integralWidth = width + 1;
      const localSum = integral[(bottom + 1) * integralWidth + right + 1]
        - integral[top * integralWidth + right + 1]
        - integral[(bottom + 1) * integralWidth + left]
        + integral[top * integralWidth + left];
      const area = (right - left + 1) * (bottom - top + 1);
      const localMean = localSum / area;
      const output = clamp(Math.round(222 + (grayscale[y * width + x] - localMean) * 2.7), 0, 255);
      const pixelIndex = (y * width + x) * 4;

      image.data[pixelIndex] = output;
      image.data[pixelIndex + 1] = output;
      image.data[pixelIndex + 2] = output;
      image.data[pixelIndex + 3] = 255;
    }
  }

  context.putImageData(image, 0, 0);
}

function createEnhancedPass(bitmap: ImageBitmap, crop: CropRegion): ScanSource {
  const scale = Math.min(2, ENHANCED_PASS_LONG_SIDE / Math.max(crop.width, crop.height));
  const outputWidth = Math.max(1, Math.round(crop.width * scale));
  const outputHeight = Math.max(1, Math.round(crop.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = outputWidth;
  canvas.height = outputHeight;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("This browser could not prepare the label image.");

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(
    bitmap,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    outputWidth,
    outputHeight,
  );
  locallyNormalizeContrast(context, outputWidth, outputHeight);

  return { input: canvas, crop, outputWidth, outputHeight };
}

async function prepareScanSources(file: File) {
  const bitmap = await createImageBitmap(file);
  const image = { x: 0, y: 0, width: bitmap.width, height: bitmap.height };

  try {
    const sources: ScanSource[] = [{
      input: file,
      crop: image,
      outputWidth: bitmap.width,
      outputHeight: bitmap.height,
    }];

    for (const crop of createCropRegions(bitmap.width, bitmap.height)) {
      sources.push(createEnhancedPass(bitmap, crop));
    }

    return { sources, image };
  } finally {
    bitmap.close();
  }
}

function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("This browser could not prepare the label crop."));
    }, "image/png");
  });
}

function recognizeWithFlorence(
  worker: Worker,
  images: Blob[],
  onProgress: (progress: FlorenceProgress) => void,
) {
  return new Promise<FlorenceResponse>((resolve, reject) => {
    const id = crypto.randomUUID();

    const handleMessage = (event: MessageEvent<Record<string, unknown>>) => {
      const data = event.data;
      if (data.id !== id) return;

      onProgress(data as unknown as FlorenceProgress);
      if (data.status === "complete") {
        cleanup();
        resolve({
          passes: data.passes as FlorencePass[],
          totalMs: data.totalMs as number,
        });
      } else if (data.status === "error") {
        cleanup();
        reject(new Error(String(data.message ?? "Florence-2 could not read this image.")));
      }
    };

    const handleError = (event: ErrorEvent) => {
      cleanup();
      reject(new Error(event.message || "The Florence-2 worker stopped unexpectedly."));
    };

    const cleanup = () => {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
    };

    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage({ id, type: "recognize", images });
  });
}

export default function LabelLens() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [scanMode, setScanMode] = useState<ScanMode>(WEBGPU_AVAILABLE ? "vision" : "classic");
  const [phase, setPhase] = useState<ScanPhase>("idle");
  const [message, setMessage] = useState("Choose a clear photo to begin.");
  const [isDragging, setIsDragging] = useState(false);
  const [copied, setCopied] = useState(false);
  const [passCount, setPassCount] = useState(0);
  const [modelProgress, setModelProgress] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const engineRef = useRef<OcrEngine | null>(null);
  const enginePromiseRef = useRef<Promise<OcrEngine> | null>(null);
  const florenceWorkerRef = useRef<Worker | null>(null);

  const isBusy = phase === "loading-model" || phase === "recognizing";
  const recognizedText = useMemo(
    () => result?.items.map((item) => item.text).join("\n") ?? "",
    [result],
  );

  const averageConfidence = useMemo(() => {
    if (!result?.items.length || result.engine === "vision") return 0;
    const total = result.items.reduce((sum, item) => sum + item.score, 0);
    return Math.round((total / result.items.length) * 100);
  }, [result]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      // Model sessions hold WASM and worker resources, so close them when the
      // page itself goes away. The same engine is intentionally reused between scans.
      if (engineRef.current) void engineRef.current.dispose();
      florenceWorkerRef.current?.terminate();
    };
  }, []);

  function selectFile(nextFile: File) {
    if (!nextFile.type.startsWith("image/")) {
      setPhase("error");
      setMessage("That file is not an image. Please choose a JPG, PNG, or WebP file.");
      return;
    }

    if (nextFile.size > MAX_FILE_SIZE) {
      setPhase("error");
      setMessage("That image is over 20 MB. Try a smaller photo.");
      return;
    }

    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setCopied(false);
    setPassCount(0);
    setModelProgress(0);
    setPhase("idle");
    setMessage("Image ready. Run OCR when you’re happy with the framing.");
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) selectFile(selected);

    // Allow choosing the same image again after a reset or failed scan.
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);

    const dropped = event.dataTransfer.files?.[0];
    if (dropped) selectFile(dropped);
  }

  async function getEngine() {
    if (engineRef.current) return engineRef.current;
    if (enginePromiseRef.current) return enginePromiseRef.current;

    // PaddleOCR is a browser-only dependency, so it is loaded on demand rather
    // than bundled into the initial page. The model downloads only on first use.
    enginePromiseRef.current = (async () => {
      const { PaddleOCR } = await import("@paddleocr/paddleocr-js");
      const engine = (await PaddleOCR.create({
        lang: "en",
        ocrVersion: "PP-OCRv6",
        worker: true,
        ortOptions: {
          backend: "auto",
          // Pin WASM loading to the package CDN. This avoids shipping a second,
          // 27 MB copy of ONNX Runtime inside the application bundle.
          wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/",
        },
      })) as OcrEngine;

      engineRef.current = engine;
      return engine;
    })();

    try {
      return await enginePromiseRef.current;
    } finally {
      enginePromiseRef.current = null;
    }
  }

  async function runOcr() {
    if (!file || isBusy) return;

    setResult(null);
    setCopied(false);
    setModelProgress(0);

    try {
      if (scanMode === "vision") {
        if (!WEBGPU_AVAILABLE) {
          throw new Error("Vision OCR needs WebGPU in this browser. Switch to Classic OCR or use a current WebGPU-capable browser.");
        }

        setPhase("loading-model");
        setMessage("Preparing Florence-2. The first scan downloads about 340 MB once.");

        const prepared = await prepareScanSources(file);
        const sources = prepared.sources.slice(1);
        const images = await Promise.all(sources.map((source) => canvasToBlob(source.input as HTMLCanvasElement)));
        setPassCount(sources.length);

        florenceWorkerRef.current ??= new Worker(new URL("./florence.worker.ts", import.meta.url), {
          type: "module",
        });

        const response = await recognizeWithFlorence(
          florenceWorkerRef.current,
          images,
          (progress) => {
            if (progress.status === "loading") {
              setPhase("loading-model");
              if (progress.message) setMessage(progress.message);
            } else if (progress.status === "model-progress") {
              const percentage = progress.progress?.progress;
              if (typeof percentage === "number") setModelProgress(Math.round(percentage));
            } else if (progress.status === "recognizing") {
              setPhase("recognizing");
              if (progress.message) setMessage(progress.message);
            }
          },
        );
        const nextResult = mergeFlorenceResults(
          response.passes,
          sources,
          prepared.image,
          response.totalMs,
        );

        setResult(nextResult);
        setPhase("complete");
        setMessage(
          nextResult.items.length
            ? `Found ${nextResult.items.length} text ${nextResult.items.length === 1 ? "line" : "lines"} with vision OCR.`
            : "No text was detected. Try moving closer or reducing glare.",
        );
        return;
      }

      if (!engineRef.current) {
        setPhase("loading-model");
        setMessage("Downloading the OCR model. The first scan takes a little longer.");
      }

      const engine = await getEngine();
      setPhase("recognizing");
      setMessage("Enhancing contrast and scanning overlapping label sections…");

      const { sources, image } = await prepareScanSources(file);
      setPassCount(sources.length);
      const scanResults = await engine.predict(
        sources.map((source) => source.input),
        LABEL_OCR_PARAMS,
      );
      const nextResult = mergeScanResults(scanResults, sources, image);

      setResult(nextResult);
      setPhase("complete");
      setMessage(
        nextResult.items.length
          ? `Found ${nextResult.items.length} text ${nextResult.items.length === 1 ? "line" : "lines"}.`
          : "No text was detected. Try moving closer or reducing glare.",
      );
    } catch (error) {
      setPhase("error");
      setMessage(readableError(error));
    }
  }

  async function copyText() {
    if (!recognizedText) return;

    try {
      await navigator.clipboard.writeText(recognizedText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setMessage("Your browser blocked clipboard access. You can still select the text below.");
    }
  }

  function downloadText() {
    if (!recognizedText) return;

    const blob = new Blob([recognizedText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = file?.name.replace(/\.[^.]+$/, "") || "label";

    link.href = url;
    link.download = `${baseName}-ocr.txt`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function resetScan() {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setCopied(false);
    setPassCount(0);
    setModelProgress(0);
    setPhase("idle");
    setMessage("Choose a clear photo to begin.");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Label Lens home">
          <span className="brand-mark" aria-hidden="true">LL</span>
          <span>Label Lens</span>
        </a>
        <div className="topbar-meta">
          <span className="model-chip">{scanMode === "vision" ? "Florence-2" : "PP-OCRv6"}</span>
          <span className="privacy-note">Runs locally in your browser</span>
        </div>
      </header>

      <section className="hero" id="top">
        <p className="eyebrow">Local vision OCR demo</p>
        <h1>Read the fine print.<br />Keep the image private.</h1>
        <p className="hero-copy">
          Drop in a photo of a food package, shampoo bottle, or product label.
          Label Lens finds the printed text without uploading the image.
        </p>
      </section>

      <section className="workspace" aria-label="OCR workspace">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <span className="step">01</span>
              <h2>Choose a label photo</h2>
            </div>
            <span className="format-note">JPG, PNG or WebP · 20 MB max</span>
          </div>

          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            accept="image/jpeg,image/png,image/webp,image/*"
            onChange={handleFileInput}
          />
          <input
            ref={cameraInputRef}
            className="visually-hidden"
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileInput}
          />

          {previewUrl ? (
            <div className="image-stage">
              {/* A standard img element preserves the source at full resolution for OCR. */}
              <img src={previewUrl} alt="Product label selected for OCR" />
              <div className="image-actions">
                <button className="text-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={isBusy}>
                  Replace image
                </button>
                <button className="text-button danger" type="button" onClick={resetScan} disabled={isBusy}>
                  Remove
                </button>
              </div>
            </div>
          ) : (
            <div
              className={`drop-zone${isDragging ? " is-dragging" : ""}`}
              onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <span className="drop-icon" aria-hidden="true">+</span>
              <strong>Drop an image here</strong>
              <span>or choose how you want to add one</span>
              <div className="picker-actions">
                <button className="primary-button compact" type="button" onClick={() => fileInputRef.current?.click()}>
                  Browse files
                </button>
                <button className="secondary-button compact" type="button" onClick={() => cameraInputRef.current?.click()}>
                  Use camera
                </button>
              </div>
            </div>
          )}

          <div className="capture-tips">
            <span>For sharper results</span>
            <p>Avoid glare, fill the frame, and keep small text in focus.</p>
          </div>

          <div className="mode-picker" aria-label="OCR engine">
            <button
              className={scanMode === "vision" ? "active" : ""}
              type="button"
              onClick={() => setScanMode("vision")}
              disabled={isBusy || !WEBGPU_AVAILABLE}
              title={!WEBGPU_AVAILABLE ? "WebGPU is not available in this browser" : undefined}
            >
              <strong>Vision OCR</strong>
              <span>Best accuracy</span>
            </button>
            <button
              className={scanMode === "classic" ? "active" : ""}
              type="button"
              onClick={() => setScanMode("classic")}
              disabled={isBusy}
            >
              <strong>Classic OCR</strong>
              <span>Faster fallback</span>
            </button>
          </div>

          <button className="primary-button scan-button" type="button" onClick={runOcr} disabled={!file || isBusy}>
            {phase === "loading-model"
              ? modelProgress > 0 ? `Loading model · ${modelProgress}%` : "Loading OCR model…"
              : phase === "recognizing" ? "Reading label…" : `Run ${scanMode === "vision" ? "vision" : "classic"} OCR`}
          </button>

          <p className="first-run-note">
            {scanMode === "vision"
              ? "Florence-2 uses WebGPU and a ~340 MB one-time model download on most GPUs. Images never leave your browser."
              : "Classic mode uses enhanced overlapping crops. It is lighter, but less accurate on faint print."}
          </p>
        </div>

        <div className="panel result-panel">
          <div className="panel-heading result-heading">
            <div>
              <span className="step">02</span>
              <h2>Recognized text</h2>
            </div>
            <span className={`status-pill ${phase}`}>
              <span className="status-dot" aria-hidden="true" />
              {phaseCopy[phase]}
            </span>
          </div>

          <div className="status-message" aria-live="polite">
            <span>{message}</span>
            {result?.items.length ? (
              <span>
                {result.engine === "classic" ? `${averageConfidence}% average confidence · ` : "Florence-2 · "}
                {passCount}-pass scan · {Math.round(result.metrics.totalMs)} ms
              </span>
            ) : null}
          </div>

          {isBusy ? (
            <div className="busy-result" role="status">
              <span className="spinner" aria-hidden="true" />
              <strong>{phase === "loading-model" ? `Preparing ${scanMode === "vision" ? "Florence-2" : "PaddleOCR"}` : "Inspecting the label"}</strong>
              <p>{phase === "loading-model" ? "The model is cached by your browser after it downloads." : "Vision OCR reads overlapping sections and may take a minute."}</p>
            </div>
          ) : result?.items.length ? (
            <div className="result-content">
              <div className="result-toolbar">
                <span>{result.items.length} detected {result.items.length === 1 ? "line" : "lines"}</span>
                <div>
                  <button className="text-button" type="button" onClick={copyText}>{copied ? "Copied" : "Copy all"}</button>
                  <button className="text-button" type="button" onClick={downloadText}>Download .txt</button>
                </div>
              </div>

              <ol className="result-list">
                {result.items.map((item, index) => (
                  <li key={`${item.text}-${index}`}>
                    <span className="line-number">{String(index + 1).padStart(2, "0")}</span>
                    <p>{item.text}</p>
                    {result.engine === "classic" ? (
                      <span className={`confidence ${item.score < 0.7 ? "low" : ""}`}>{Math.round(item.score * 100)}%</span>
                    ) : (
                      <span className="confidence vision">VLM</span>
                    )}
                  </li>
                ))}
              </ol>
            </div>
          ) : (
            <div className="empty-result">
              <span className="result-number">A</span>
              <p>{phase === "error" ? "Check the note above, then try again." : "Your extracted label text will appear here in reading order, ready to copy or download."}</p>
            </div>
          )}
        </div>
      </section>

      <section className="how-it-works" aria-labelledby="how-title">
        <div>
          <p className="eyebrow">What happens locally</p>
          <h2 id="how-title">Two engines. One private workflow.</h2>
        </div>
        <ol>
          <li><span>1</span><p><strong>Enhancement</strong> enlarges small print and evens out shadows on curved labels.</p></li>
          <li><span>2</span><p><strong>Vision OCR</strong> uses language context to recover difficult words and full lines.</p></li>
          <li><span>3</span><p><strong>Classic OCR</strong> remains available for lighter devices without WebGPU.</p></li>
        </ol>
      </section>

      <footer>
        <span>Powered by Florence-2, Transformers.js &amp; PaddleOCR.js</span>
        <span>English / Latin labels · No account · No image upload</span>
      </footer>
    </main>
  );
}
