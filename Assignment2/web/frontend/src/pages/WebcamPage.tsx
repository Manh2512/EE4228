import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function WebcamPage() {
  const navigate    = useNavigate();
  const videoRef    = useRef<HTMLVideoElement>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const wsRef       = useRef<WebSocket | null>(null);
  const pausedRef   = useRef(false);
  const rafRef      = useRef<number>(0);
  const activeRef   = useRef(true);
  const lastSendRef = useRef<number>(0);
  const streamRef   = useRef<MediaStream | null>(null);
  const INTERVAL_TIME_MS = 200; // send frame each 100ms

  const [paused,     setPaused]     = useState(false);
  const [frameData,  setFrameData]  = useState<string>("");
  const [status,     setStatus]     = useState("Connecting…");

  useEffect(() => {
    activeRef.current = true;
    const ws = new WebSocket(`ws://${window.location.host}/ws/webcam`);
    wsRef.current = ws;

    ws.onopen  = () => setStatus("Live");
    ws.onerror = () => setStatus("WebSocket error");
    ws.onclose = () => setStatus("Disconnected");

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data as string);
      setFrameData(`data:image/jpeg;base64,${data.annotated_frame}`);
    };

    // Start webcam
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((s) => {
        // Stop any previously opened stream (StrictMode double-invoke guard)
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = s;
        if (activeRef.current && videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.play();
          rafRef.current = requestAnimationFrame(captureLoop);
        } else {
          s.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
      })
      .catch(() => setStatus("Camera access denied"));

    return () => {
      activeRef.current = false;
      cancelAnimationFrame(rafRef.current);
      ws.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  function captureLoop() {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    const ws     = wsRef.current;

    if (video && canvas && ws && ws.readyState === WebSocket.OPEN) {
      const now = performance.now();
      if (!pausedRef.current && video.videoWidth > 0 && now-lastSendRef.current >= INTERVAL_TIME_MS) {
        lastSendRef.current = now;
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d")!.drawImage(video, 0, 0);
        canvas.toBlob(
          (blob) => { if (blob) ws.send(blob); },
          "image/jpeg",
          0.8
        );
      }
    }
    if (activeRef.current) {
      rafRef.current = requestAnimationFrame(captureLoop);
    }
  }

  function togglePause() {
    const next = !paused;
    setPaused(next);
    pausedRef.current = next;
    wsRef.current?.send(JSON.stringify({ action: next ? "pause" : "resume" }));
  }

  return (
    <div className="page">
      <div className="page-title">EE4228 GUI Demo</div>
      <button className="back-link" onClick={() => navigate("/")}>◀ Back</button>

      {/* Hidden raw video + offscreen canvas for frame capture */}
      <video ref={videoRef} style={{ display: "none" }} muted playsInline />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div className="webcam-wrapper">
        <div className="panel-label">Webcam</div>
        <div className="webcam-box">
          {frameData
            ? <img src={frameData} alt="annotated feed" />
            : <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>{status}</span>
          }
        </div>
      </div>

      <div className="webcam-controls">
        <button className="pause-btn" onClick={togglePause}>
          {paused ? "Continue" : "Pause"}
        </button>
      </div>

      <p className="status">{status}</p>
    </div>
  );
}
