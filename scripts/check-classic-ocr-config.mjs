import { readFile } from "node:fs/promises";
import { PaddleOCR } from "@paddleocr/paddleocr-js";

const configUrl = new URL("../app/classic-ocr-config.json", import.meta.url);
const config = JSON.parse(await readFile(configUrl, "utf8"));

// Model selection is resolved before Paddle downloads anything. This catches
// invalid language/version pairs quickly in CI without fetching model weights.
const engine = await PaddleOCR.create({ ...config, initialize: false });
await engine.dispose();

console.log(`Classic OCR config is valid (${config.lang}, ${config.ocrVersion}).`);
