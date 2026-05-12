function MobileCameraView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // 1. 카메라 즉시 실행 (아이폰 Safari 팝업 유도)
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: 640, height: 480 },
          audio: false
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
        
        // 2. 카메라가 켜지면 그제서야 WebSocket 연결 (순서가 중요!)
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsRef.current = new WebSocket(`${protocol}//${window.location.host}/ws`);
        wsRef.current.onopen = () => setIsConnected(true);
        wsRef.current.onclose = () => setIsConnected(false);

        const sendLoop = () => {
          if (wsRef.current?.readyState === 1 && videoRef.current?.readyState === 4) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0, 640, 480);
            canvas.toBlob(blob => wsRef.current.send(blob), 'image/jpeg', 0.5);
          }
          setTimeout(sendLoop, 100); // 부하를 줄이기 위해 10FPS로 설정
        };
        sendLoop();
      } catch (err) {
        alert("카메라 권한을 '허용'해야 합니다. 설정에서 확인해 주세요.");
      }
    };

    startCamera();
  }, []);

  return (
    <div className="h-screen bg-black flex flex-col items-center justify-center text-white">
      <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: 'auto' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
      <div className={`mt-4 px-4 py-2 rounded-full font-bold ${isConnected ? 'bg-green-600' : 'bg-red-600'}`}>
        {isConnected ? '● PC 전송 중' : '서버 연결 대기 중...'}
      </div>
    </div>
  );
}