import { BrowserRouter, Routes, Route } from "react-router-dom";
import MainPage from "./pages/MainPage";
import ImagePage from "./pages/ImagePage";
import VideoPage from "./pages/VideoPage";
import WebcamPage from "./pages/WebcamPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/image" element={<ImagePage />} />
        <Route path="/video" element={<VideoPage />} />
        <Route path="/webcam" element={<WebcamPage />} />
      </Routes>
    </BrowserRouter>
  );
}
