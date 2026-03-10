import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function ImagePage() {
  const navigate = useNavigate();
  const inputRef  = useRef<HTMLInputElement>(null);
  const [rawSrc,       setRawSrc]       = useState<string>("");
  const [processedSrc, setProcessedSrc] = useState<string>("");
  const [status,       setStatus]       = useState<string>("");

  async function handleFile(file: File) {
    setRawSrc(URL.createObjectURL(file));
    setProcessedSrc("");
    setStatus("Processing…");

    const form = new FormData();
    form.append("image", file);

    try {
      const res  = await fetch("/api/process/image", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? "Server error");
      setProcessedSrc(`data:image/jpeg;base64,${data.processed_image}`);
      setStatus(`${data.face_count} face(s) detected`);
    } catch (err: unknown) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  return (
    <div className="page">
      <div className="page-title">EE4228 GUI Demo</div>
      <button className="back-link" onClick={() => navigate("/")}>◀ Back</button>

      <div className="upload-row">
        <label className="upload-btn">
          Upload an Image
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </label>
      </div>

      <div className="panels">
        <div>
          <div className="panel-label">Raw Image</div>
          <div className="panel-box">
            {rawSrc && <img src={rawSrc} alt="raw" />}
          </div>
        </div>
        <div>
          <div className="panel-label">Processed Image</div>
          <div className="panel-box">
            {processedSrc && <img src={processedSrc} alt="processed" />}
          </div>
        </div>
      </div>

      {status && <p className="status">{status}</p>}
    </div>
  );
}
