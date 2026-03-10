import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function VideoPage() {
  const navigate    = useNavigate();
  const inputRef    = useRef<HTMLInputElement>(null);
  const [rawSrc,       setRawSrc]       = useState<string>("");
  const [processedSrc, setProcessedSrc] = useState<string>("");
  const [status,       setStatus]       = useState<string>("");

  async function handleFile(file: File) {
    setRawSrc(URL.createObjectURL(file));
    setProcessedSrc("");
    setStatus("Uploading…");

    const form = new FormData();
    form.append("video", file);

    try {
      const res        = await fetch("/api/process/video", { method: "POST", body: form });
      const { job_id } = await res.json();
      setStatus("Processing…");
      await pollUntilDone(job_id);
    } catch (err: unknown) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function pollUntilDone(jobId: string) {
    while (true) {
      const res    = await fetch(`/api/process/video/status/${jobId}`);
      const data   = await res.json();

      if (data.status === "done") {
        setProcessedSrc(`/api/process/video/result/${jobId}`);
        setStatus("Done");
        break;
      }
      if (data.status === "error") {
        setStatus(`Processing failed: ${data.detail ?? "unknown error"}`);
        break;
      }

      setStatus(`Processing… ${data.progress ?? 0}%`);
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  return (
    <div className="page">
      <div className="page-title">EE4228 GUI Demo</div>
      <button className="back-link" onClick={() => navigate("/")}>◀ Back</button>

      <div className="upload-row">
        <label className="upload-btn">
          Upload a Video
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </label>
      </div>

      <div className="panels">
        <div>
          <div className="panel-label">Raw Video</div>
          <div className="panel-box">
            {rawSrc && <video src={rawSrc} controls />}
          </div>
        </div>
        <div>
          <div className="panel-label">Processed Video</div>
          <div className="panel-box">
            {processedSrc && <video src={processedSrc} controls />}
          </div>
        </div>
      </div>

      {status && <p className="status">{status}</p>}
    </div>
  );
}
