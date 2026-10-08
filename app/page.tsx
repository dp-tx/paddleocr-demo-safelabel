import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import type { OcrResult } from "@paddleocr/paddleocr-js";

type ScanPhase = "idle" | "loading-model" | "recognizing" | "complete" | "error";

// Keeping the engine behind a small interface makes the page easier to read and
// prevents browser-only implementation details from leaking into the UI code.
type OcrEngine = {
  predict: (input: unknown) => Promise<OcrResult[]>;
  dispose: () => Promise<void>;
};

const MAX_FILE_SIZE = 20 * 1024 * 1024;

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

export default function LabelLens() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<OcrResult | null>(null);
  const [phase, setPhase] = useState<ScanPhase>("idle");
  const [message, setMessage] = useState("Choose a clear photo to begin.");
  const [isDragging, setIsDragging] = useState(false);
  const [copied, setCopied] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const engineRef = useRef<OcrEngine | null>(null);
  const enginePromiseRef = useRef<Promise<OcrEngine> | null>(null);

  const isBusy = phase === "loading-model" || phase === "recognizing";
  const recognizedText = useMemo(
    () => result?.items.map((item) => item.text).join("\n") ?? "",
    [result],
  );

  const averageConfidence = useMemo(() => {
    if (!result?.items.length) return 0;
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

    try {
      if (!engineRef.current) {
        setPhase("loading-model");
        setMessage("Downloading the OCR model. The first scan takes a little longer.");
      }

      const engine = await getEngine();
      setPhase("recognizing");
      setMessage("Looking for text and reading each detected line…");

      const [nextResult] = await engine.predict(file);
      if (!nextResult) throw new Error("PaddleOCR did not return a result.");

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
          <span className="model-chip">PP-OCRv6</span>
          <span className="privacy-note">Runs locally in your browser</span>
        </div>
      </header>

      <section className="hero" id="top">
        <p className="eyebrow">PaddleOCR browser demo</p>
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

          <button className="primary-button scan-button" type="button" onClick={runOcr} disabled={!file || isBusy}>
            {phase === "loading-model" ? "Loading OCR model…" : phase === "recognizing" ? "Reading label…" : "Run OCR"}
          </button>

          <p className="first-run-note">The first scan downloads the free OCR model. Later scans reuse it.</p>
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
              <span>{averageConfidence}% average confidence · {Math.round(result.metrics.totalMs)} ms</span>
            ) : null}
          </div>

          {isBusy ? (
            <div className="busy-result" role="status">
              <span className="spinner" aria-hidden="true" />
              <strong>{phase === "loading-model" ? "Preparing PaddleOCR" : "Inspecting the label"}</strong>
              <p>{phase === "loading-model" ? "This one-time download can take a moment." : "Small print and curved labels may take a few seconds."}</p>
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
                    <span className={`confidence ${item.score < 0.7 ? "low" : ""}`}>{Math.round(item.score * 100)}%</span>
                  </li>
                ))}
              </ol>
            </div>
          ) : (
            <div className="empty-result">
              <span className="result-number">A</span>
              <p>{phase === "error" ? "Check the note above, then try again." : "Your extracted label text will appear here, with confidence scores for every detected line."}</p>
            </div>
          )}
        </div>
      </section>

      <section className="how-it-works" aria-labelledby="how-title">
        <div>
          <p className="eyebrow">What happens locally</p>
          <h2 id="how-title">One image. Two neural models.</h2>
        </div>
        <ol>
          <li><span>1</span><p><strong>Detection</strong> finds text regions at different sizes and angles.</p></li>
          <li><span>2</span><p><strong>Recognition</strong> turns each detected crop into editable text.</p></li>
          <li><span>3</span><p><strong>Confidence</strong> highlights lines worth checking manually.</p></li>
        </ol>
      </section>

      <footer>
        <span>Powered by PaddleOCR.js</span>
        <span>English / Latin labels · No account · No image upload</span>
      </footer>
    </main>
  );
}
