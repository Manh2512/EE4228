import { Link } from "react-router-dom";

const modes = [
  {
    to: "/image",
    icon: "/image-icon.png",
    label: "Image",
    desc: "Detect and recognize faces within an image",
  },
  {
    to: "/video",
    icon: "/video-icon.png",
    label: "Video",
    desc: "Detect and recognize faces within a video",
  },
  {
    to: "/webcam",
    icon: "/webcam-icon.png",
    label: "Webcam",
    desc: "Real-time detect and recognize faces with webcam",
  },
];

export default function MainPage() {
  return (
    <div className="page">
      <div className="page-title">EE4228 GUI Demo</div>
      <div className="cards">
        {modes.map((m) => (
          <Link key={m.to} to={m.to} className="card">
            <img src={m.icon} alt={m.label} className="card-icon-img" />
            <span className="card-label">{m.label}</span>
            <span className="card-desc">{m.desc}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
