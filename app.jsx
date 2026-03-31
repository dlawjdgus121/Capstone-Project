import React, { useState, useEffect, useRef } from 'react';
import { Camera, Upload, CheckCircle, Play, Square, Loader2, ChevronRight, Smartphone } from 'lucide-react';

export default function App() {
  const [mode, setMode] = useState(window.location.pathname.includes('mobile') ? 'mobile' : 'pc');
  const [view, setView] = useState('landing'); // landing, upload, results, camera
  const [manualSteps, setManualSteps] = useState([]);
  const [status, setStatus] = useState({ current_idx: 0, ai_response: '대기 중...', is_active: false, has_frame: false });
  
  // PC용 상태 폴링
  useEffect(() => {
    if (mode === 'pc') {
      const timer = setInterval(async () => {
        try {
          const res = await fetch('/status');
          const data = await res.json();
          setStatus(data);
        } catch (e) {}
      }, 2000);
      return () => clearInterval(timer);
    }
  }, [mode]);

  if (mode === 'mobile') {
    return <MobileCameraView />;
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased">
      {/* 1. Landing */}
      {view === 'landing' && (
        <div className="flex flex-col items-center justify-center min-h-screen p-6 text-center">
          <h1 className="text-6xl font-black mb-4 tracking-tighter">AI ASSEMBLY COACH</h1>
          <p className="text-slate-500 mb-12 text-xl max-w-lg">PC에서 매뉴얼을 업로드하고,<br/>모바일 카메라로 조립을 시작하세요.</p>
          <div className="flex gap-4">
            <button onClick={() => setView('upload')} className="bg-blue-600 text-white px-10 py-5 rounded-full font-bold text-xl flex items-center gap-3">
              시작하기 <ChevronRight size={28} />
            </button>
          </div>
          <div className="mt-12 p-6 bg-blue-50 rounded-3xl border border-blue-100 inline-flex items-center gap-4">
            <div className="bg-white p-3 rounded-2xl shadow-sm"><Smartphone className="text-blue-600" /></div>
            <div className="text-left">
              <p className="text-xs font-black text-blue-600 uppercase">Mobile 필수</p>
              <p className="text-sm text-slate-600 font-bold">스마트폰으로도 접속하여 카메라를 켜주세요.</p>
            </div>
          </div>
        </div>
      )}

      {/* 2. Upload & 3. Results (생략 - 이전 App.jsx와 동일한 로직 적용) */}
      {/* ... (생략된 코드는 이전 App.jsx의 handleFileUpload, renderSteps 로직과 동일) ... */}

      {/* 4. Coaching View (PC Monitoring) */}
      {view === 'camera' && (
        <div className="h-screen flex">
          <div className="w-1/3 bg-white border-r p-8 overflow-y-auto">
            <h3 className="font-black text-2xl mb-8">조립 가이드</h3>
            {/* manualSteps 리스트 렌더링 */}
          </div>
          <div className="flex-1 bg-black relative">
            {/* 서버에서 쏘는 MJPEG 스트림 수신 */}
            <img src="/stream" className="w-full h-full object-contain" alt="Live from Mobile" />
            
            <div className="absolute top-8 left-8 right-8 bg-white/90 p-8 rounded-3xl shadow-2xl border-l-[12px] border-blue-500">
              <p className="text-2xl font-bold">{status.ai_response}</p>
              {!status.has_frame && <p className="text-red-500 mt-2">⚠️ 모바일 카메라가 연결되지 않았습니다.</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// --- 모바일 전송 전용 컴포넌트 ---
function MobileCameraView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    wsRef.current = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current.onopen = () => setIsConnected(true);

    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } }
    }).then(stream => {
      videoRef.current.srcObject = stream;
      const sendLoop = () => {
        if (wsRef.current?.readyState === 1 && videoRef.current?.readyState === 4) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          canvas.width = 640; canvas.height = 480; // 모바일 전송 부하 감소
          ctx.drawImage(videoRef.current, 0, 0, 640, 480);
          canvas.toBlob(blob => wsRef.current.send(blob), 'image/jpeg', 0.5);
        }
        setTimeout(sendLoop, 50); // 20FPS
      };
      sendLoop();
    });
  }, []);

  return (
    <div className="h-screen bg-black flex flex-col items-center justify-center p-6 text-white text-center">
      <video ref={videoRef} autoPlay playsInline muted className="hidden" />
      <canvas ref={canvasRef} className="hidden" />
      <div className={`w-20 h-20 rounded-full mb-6 ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
      <h1 className="text-2xl font-black mb-2">{isConnected ? 'PC로 전송 중' : '서버 연결 중...'}</h1>
      <p className="text-zinc-500">PC 화면의 대시보드를 확인하면서<br/>조립을 진행해 주세요.</p>
    </div>
  );
}