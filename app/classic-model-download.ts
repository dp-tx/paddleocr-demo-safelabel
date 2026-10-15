type ModelSource = {
  label: string;
  url: string;
};

export type ModelDownloadProgress = {
  loadedBytes: number;
  totalBytes: number | null;
  percent: number | null;
};

export type DownloadedClassicModels = {
  detectionUrl: string;
  recognitionUrl: string;
  revoke: () => void;
};

function readContentLength(response: Response) {
  const value = response.headers.get("content-length");
  if (value === null) return null;

  const bytes = Number(value);
  return Number.isFinite(bytes) && bytes > 0 ? bytes : null;
}

async function readResponseBlob(
  response: Response,
  onChunk: (bytes: number) => void,
  signal: AbortSignal,
) {
  // Older browsers may not expose a readable response stream. They can still
  // load the model; the UI simply stays indeterminate until that file finishes.
  if (!response.body) {
    const blob = await response.blob();
    onChunk(blob.size);
    return blob;
  }

  const reader = response.body.getReader();
  const chunks: BlobPart[] = [];
  const cancelRead = () => {
    // If the companion model fails, stop this stream instead of continuing a
    // large download that the current scan can no longer use.
    void reader.cancel(signal.reason).catch(() => undefined);
  };

  signal.addEventListener("abort", cancelRead, { once: true });

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;

      chunks.push(value as BlobPart);
      onChunk(value.byteLength);
    }
  } finally {
    signal.removeEventListener("abort", cancelRead);
    reader.releaseLock();
  }

  return new Blob(chunks, { type: "application/x-tar" });
}

/**
 * Downloads both PaddleOCR model archives while reporting their combined byte
 * progress. Blob URLs let the existing worker consume these exact bytes, which
 * keeps inference off the main thread and avoids a second network download.
 */
export async function downloadClassicModels(
  sources: readonly [ModelSource, ModelSource],
  onProgress: (progress: ModelDownloadProgress) => void,
): Promise<DownloadedClassicModels> {
  const controller = new AbortController();

  try {
    const downloads = await Promise.all(
      sources.map(async (source) => {
        const response = await fetch(source.url, {
          cache: "force-cache",
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Failed to download ${source.label}: HTTP ${response.status}.`);
        }

        return {
          response,
          totalBytes: readContentLength(response),
        };
      }),
    );

    const knownTotal = downloads.every((download) => download.totalBytes !== null)
      ? downloads.reduce((sum, download) => sum + (download.totalBytes ?? 0), 0)
      : null;
    const loadedByFile = downloads.map(() => 0);
    let lastReportedBytes = -1;
    let lastReportedPercent: number | null | undefined;

    const reportProgress = (force = false) => {
      const loadedBytes = loadedByFile.reduce((sum, bytes) => sum + bytes, 0);
      const percent = knownTotal === null
        ? null
        : Math.min(100, Math.floor((loadedBytes / knownTotal) * 100));

      // Network streams can yield thousands of small chunks. Updating at most
      // once per percentage point keeps slower phones responsive while remaining
      // precise; without a known total, update every quarter megabyte instead.
      if (!force) {
        if (percent !== null && percent === lastReportedPercent) return;
        if (percent === null && loadedBytes - lastReportedBytes < 256 * 1024) return;
      }

      lastReportedBytes = loadedBytes;
      lastReportedPercent = percent;
      onProgress({ loadedBytes, totalBytes: knownTotal, percent });
    };

    reportProgress(true);

    const blobs = await Promise.all(
      downloads.map((download, index) => readResponseBlob(
        download.response,
        (bytes) => {
          loadedByFile[index] += bytes;
          reportProgress();
        },
        controller.signal,
      )),
    );

    // If Content-Length was unavailable, the completed blobs give us an exact
    // final total even though intermediate progress had to remain indeterminate.
    const finalTotal = knownTotal ?? blobs.reduce((sum, blob) => sum + blob.size, 0);
    onProgress({ loadedBytes: finalTotal, totalBytes: finalTotal, percent: 100 });

    const objectUrls: string[] = [];

    try {
      objectUrls.push(URL.createObjectURL(blobs[0]), URL.createObjectURL(blobs[1]));
    } catch (error) {
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
      throw error;
    }

    let revoked = false;

    return {
      detectionUrl: objectUrls[0],
      recognitionUrl: objectUrls[1],
      revoke: () => {
        if (revoked) return;
        revoked = true;
        objectUrls.forEach((url) => URL.revokeObjectURL(url));
      },
    };
  } catch (error) {
    controller.abort(error);
    throw error;
  }
}
