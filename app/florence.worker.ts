/// <reference lib="webworker" />

import {
  AutoProcessor,
  AutoTokenizer,
  Florence2ForConditionalGeneration,
  RawImage,
} from "@huggingface/transformers";

const MODEL_ID = "onnx-community/Florence-2-base-ft";
const OCR_TASK = "<OCR_WITH_REGION>";

type ModelBundle = Awaited<ReturnType<typeof loadModel>>;

let modelPromise: Promise<ModelBundle> | null = null;

async function supportsFloat16() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter?.features.has("shader-f16") ?? false;
  } catch {
    return false;
  }
}

async function loadModel(requestId: string) {
  self.postMessage({
    id: requestId,
    status: "loading",
    message: "Downloading Florence-2 (about 340 MB, cached after the first scan)…",
  });

  const useFloat16 = await supportsFloat16();
  const progressCallback = (progress: Record<string, unknown>) => {
    self.postMessage({ id: requestId, status: "model-progress", progress });
  };

  // Keeping the vision encoder at full half precision matters for small print.
  // The language encoder/decoder tolerate 4-bit weights well and save hundreds
  // of megabytes, matching Hugging Face's recommended Florence-2 WebGPU setup.
  const [model, tokenizer, processor] = await Promise.all([
    Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: {
        embed_tokens: useFloat16 ? "fp16" : "fp32",
        vision_encoder: useFloat16 ? "fp16" : "fp32",
        encoder_model: "q4",
        decoder_model_merged: "q4",
      },
      device: "webgpu",
      progress_callback: progressCallback,
    }),
    AutoTokenizer.from_pretrained(MODEL_ID),
    AutoProcessor.from_pretrained(MODEL_ID),
  ]);

  return { model, tokenizer, processor };
}

async function recognize(requestId: string, images: Blob[]) {
  if (!("gpu" in navigator)) {
    throw new Error("Vision OCR needs WebGPU. Use a current Chrome, Edge, Firefox, or Safari release, or switch to Classic OCR.");
  }

  // The first request creates the model promise; later scans reuse the same
  // compiled sessions and the browser's persistent model cache.
  modelPromise ??= loadModel(requestId);
  const { model, tokenizer, processor } = await modelPromise;

  self.postMessage({
    id: requestId,
    status: "recognizing",
    message: "Reading each label section with Florence-2…",
  });

  const startedAt = performance.now();
  const passes = [];

  for (let index = 0; index < images.length; index += 1) {
    self.postMessage({
      id: requestId,
      status: "recognizing",
      message: `Reading label section ${index + 1} of ${images.length}…`,
    });

    const image = await RawImage.fromBlob(images[index]);
    const visionInputs = await processor(image);
    const prompts = processor.construct_prompts(OCR_TASK);
    const textInputs = tokenizer(prompts);

    const generatedIds = await model.generate({
      ...textInputs,
      ...visionInputs,
      max_new_tokens: 512,
      num_beams: 2,
      do_sample: false,
    });
    const generatedText = tokenizer.batch_decode(generatedIds, {
      skip_special_tokens: false,
    })[0];
    const parsed = processor.post_process_generation(generatedText, OCR_TASK, image.size);
    const ocr = parsed[OCR_TASK];

    if (typeof ocr === "string") {
      passes.push({ labels: [ocr], quadBoxes: [], width: image.width, height: image.height });
    } else {
      passes.push({
        labels: ocr.labels.map(String),
        quadBoxes: (ocr.quad_boxes ?? []) as number[][],
        width: image.width,
        height: image.height,
      });
    }
  }

  self.postMessage({
    id: requestId,
    status: "complete",
    passes,
    totalMs: performance.now() - startedAt,
  });
}

self.addEventListener("message", (event: MessageEvent<{ id: string; type: string; images: Blob[] }>) => {
  const { id, type, images } = event.data;
  if (type !== "recognize") return;

  void recognize(id, images).catch((error: unknown) => {
    // Allow a failed model load to be retried after the user changes browser
    // settings or switches devices.
    modelPromise = null;
    self.postMessage({
      id,
      status: "error",
      message: error instanceof Error ? error.message : "Florence-2 could not read this image.",
    });
  });
});

export {};
