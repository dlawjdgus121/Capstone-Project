import React, { useState } from 'react';
import CardSwap, { Card } from './CardSwap';
import './App.css';
import {
  BsCodeSlash,
  BsSliders,
  BsCircleFill,
  BsCloudUpload,
  BsFileEarmarkPdf,
  BsFileEarmarkImage,
  BsHouseDoorFill,
  BsCameraFill,
  BsFilm,           
  BsLink45Deg,      
  BsYoutube,
  BsTrash            // [NEW] 삭제(휴지통) 아이콘 추가
} from 'react-icons/bs';

import reliableImg from './img/image_Realiable.png';
import smoothImg from './img/image_Smooth.png';
import customizableImg from './img/image_Customizable.png';

import * as pdfjsLib from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

const API = "http://localhost:8005";

function App() {
  const [page, setPage] = useState("landing");
  const [isActive, setIsActive] = useState(false);
  
  // 파일/링크 관련 상태
  const [uploadedFile, setUploadedFile] = useState(null);
  const [linkUrl, setLinkUrl] = useState(""); 
  const [mediaType, setMediaType] = useState(null); 

  // 뷰어 관련 상태
  const [previewSrc, setPreviewSrc] = useState(null); 
  const [pdfPage, setPdfPage] = useState(0);

  const [cameraOn, setCameraOn] = useState(false);
  const [streamTs, setStreamTs] = useState(0);

  const [confirmExit, setConfirmExit] = useState(true);
  const [showAlert, setShowAlert] = useState(true);
  const [modalType, setModalType] = useState(null);

  /* ================= HELPERS ================= */
  const getYoutubeId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  /* ================= PDF → IMAGE ================= */
  const renderPdfToImages = async (file) => {
    const pdf = await pdfjsLib.getDocument(URL.createObjectURL(file)).promise;
    const images = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 2 });
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({ canvasContext: ctx, viewport }).promise;
      images.push(canvas.toDataURL("image/png"));
    }
    setPreviewSrc(images);
    setPdfPage(0);
    setMediaType("pdf");
  };

  /* ================= INPUT HANDLING ================= */
  // 1. 파일 삭제 기능 [NEW]
  const handleRemoveFile = (e) => {
    // 부모 라벨의 클릭 이벤트(파일 선택창 열기)가 발생하지 않도록 막음
    if (e) {
      e.stopPropagation();
      e.preventDefault();
    }
    setUploadedFile(null);
    setLinkUrl("");
    setMediaType(null);
    setPreviewSrc(null);
    setPdfPage(0);
  };

  // 2. 파일 업로드 처리
  const validateFile = (file) => {
    const imageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    const videoTypes = ['video/mp4', 'video/webm', 'video/ogg'];
    const pdfType = ['application/pdf'];

    setUploadedFile(file);
    setLinkUrl(""); 

    if (pdfType.includes(file.type)) {
      renderPdfToImages(file);
    } 
    else if (videoTypes.includes(file.type)) {
      setPreviewSrc(URL.createObjectURL(file));
      setMediaType("video");
    } 
    else if (imageTypes.includes(file.type)) {
      setPreviewSrc([URL.createObjectURL(file)]);
      setPdfPage(0);
      setMediaType("image");
    } 
    else {
      alert("지원하지 않는 파일 형식입니다. (PDF, 이미지, 동영상 가능)");
      setUploadedFile(null);
    }
  };

  // 3. 링크 입력 처리
  const handleLinkSubmit = () => {
    if (!linkUrl) return;
    const videoId = getYoutubeId(linkUrl);
    
    if (videoId) {
      setMediaType("youtube");
      setPreviewSrc(videoId);
      setUploadedFile({ name: "YouTube Link", type: "link" });
    } else {
      setMediaType("video");
      setPreviewSrc(linkUrl);
      setUploadedFile({ name: "Video Link", type: "link" });
    }
  };

  /* ================= DRAG & DROP ================= */
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) validateFile(file);
  };
  const handleDragOver = (e) => { e.preventDefault(); setIsActive(true); };
  const handleDragLeave = () => setIsActive(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setIsActive(false);
    const file = e.dataTransfer.files[0];
    if (file) validateFile(file);
  };

  /* ================= SERVER ================= */
  const startCamera = async () => {
    await fetch(`${API}/start`, { method: "POST" });
    setStreamTs(Date.now());
    setCameraOn(true);
    setPage("camera");
  };

  const runVLM = async () => {
    await fetch(`${API}/vlm`, { method: "POST" });
    if (showAlert) alert("AI 코치 분석이 완료되었습니다.");
  };

  const stopCamera = async () => {
    if (confirmExit && !window.confirm("코칭을 종료하시겠습니까?")) return;
    await fetch(`${API}/stop`, { method: "POST" });
    setCameraOn(false);
    setPage("upload");
  };

  /* ================= STYLES ================= */
  // 뷰어 크기 통일 스타일 (검은 배경 + 비율 유지) [NEW]
  const viewerStyle = {
    width: '100%',
    height: '100%',
    objectFit: 'contain', // 비율 유지하며 꽉 채우기 (잘림 없음)
    backgroundColor: '#000', // 빈 공간 검은색 처리
    display: 'block'
  };

  return (
    <div className="app-screen">

      {/* ================= LANDING ================= */}
      {page === "landing" && (
        <div className="landing-container">
          <div className="text-section">
            <h1>AI 코치 서비스에<br />오신 걸 환영합니다</h1>
            <button className="btn-primary" onClick={() => setPage("home")}>시작하기</button>
          </div>
          <CardSwap width={650} height={850}>
            <Card><BsCodeSlash /><img src={reliableImg} alt="" /></Card>
            <Card><BsCircleFill /><img src={smoothImg} alt="" /></Card>
            <Card><BsSliders /><img src={customizableImg} alt="" /></Card>
          </CardSwap>
        </div>
      )}

      {/* ================= HOME ================= */}
      {page === "home" && (
        <div className="home-layout">
          <aside className="sidebar">
            <h2 className="menu-title">메뉴</h2>
            <nav className="menu-list">
              <button className="menu-btn" onClick={() => setModalType("guide")}>가이드라인</button>
              <button className="menu-btn" onClick={() => setModalType("usage")}>사용 방법</button>
              <button className="menu-btn" onClick={() => setModalType("setting")}>설정</button>
            </nav>
          </aside>
          <main className="home-content">
            <h1 className="home-title">AI 코치 홈</h1>
            <p className="home-subtitle">매뉴얼(PDF, 영상, 이미지, 링크)을 업로드하세요.</p>
            <button className="btn-white-block" onClick={() => setPage("upload")}>매뉴얼 업로드로 이동</button>
          </main>
        </div>
      )}

      {/* ================= UPLOAD ================= */}
      {page === "upload" && (
        <div className="upload-container">
          <div className="upload-card">
            <h2 className="upload-title">매뉴얼 업로드</h2>

            {/* A. 파일 드래그 앤 드롭 영역 */}
            <label
              className={`upload-area ${isActive ? 'active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              style={{ position: 'relative' }} // 삭제 버튼 배치를 위해 relative 추가
            >
              <input type="file" accept="image/*,video/*,.pdf" onChange={handleFileChange} />
              
              {uploadedFile ? (
                <div className="file-preview">
                  {/* 삭제 버튼 추가 [NEW] */}
                 <button 
  onClick={handleRemoveFile}
  className="btn-remove-file"
  style={{
    position: 'absolute',
    top: '10px',
    right: '10px',
    background: 'rgba(0,0,0,0.6)', // 반투명 검은 배경
    color: 'white',
    border: 'none',
    borderRadius: '6px',        // 동그라미(50%) 대신 둥근 사각형으로 변경
    padding: '6px 12px',        // 글자가 들어가므로 내부 여백 추가
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '6px',                 // 아이콘과 글자 사이 간격
    zIndex: 10,
    fontSize: '14px',           // 글자 크기 조정
    fontWeight: 'bold'
  }}
  title="파일 삭제"
>
  <BsTrash size={14} />         {/* 아이콘 크기 살짝 조절 */}
  <span>삭제</span>              {/* 글자 추가 */}
</button>

                  {/* 아이콘 표시 */}
                  {uploadedFile.type?.includes('pdf') && <BsFileEarmarkPdf size={48} />}
                  {uploadedFile.type?.includes('image') && <BsFileEarmarkImage size={48} />}
                  {uploadedFile.type?.includes('video') && <BsFilm size={48} />}
                  {uploadedFile.type === 'link' && <BsYoutube size={48} color="red"/>}
                  
                  <p>{uploadedFile.name}</p>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <BsCloudUpload size={50} />
                  <p>PDF, 이미지, 영상 파일 드래그</p>
                </div>
              )}
            </label>

            {/* B. 링크 입력 영역 */}
            <div style={{ margin: '15px 0', width: '100%', display: 'flex', gap: '10px' }}>
               <input 
                 type="text" 
                 placeholder="또는 유튜브/영상 링크를 입력하세요" 
                 className="link-input"
                 style={{ flex: 1, padding: '10px', borderRadius: '8px', border: '1px solid #ddd' }}
                 value={linkUrl}
                 onChange={(e) => setLinkUrl(e.target.value)}
               />
               <button 
                 onClick={handleLinkSubmit}
                 style={{ padding: '0 20px', borderRadius: '8px', border: 'none', background: '#333', color: '#fff', cursor: 'pointer' }}
               >
                 확인
               </button>
            </div>

            <div className="upload-actions">
              <button
                className={`btn-action btn-start ${!uploadedFile ? 'disabled' : ''}`}
                disabled={!uploadedFile}
                onClick={startCamera}
              >
                <BsCameraFill /> 코칭 시작
              </button>
              <button className="btn-action btn-home" onClick={() => setPage("home")}>
                <BsHouseDoorFill /> 홈으로
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ================= CAMERA ================= */}
      {page === "camera" && (
        <div className="camera-layout">
          {/* PDF Panel 스타일 개선: 배경 검정, 중앙 정렬 */}
          <div className="pdf-panel" style={{ background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
            
            {/* 1. 이미지 및 PDF 모드 */}
            {(mediaType === 'pdf' || mediaType === 'image') && previewSrc && (
              <>
                <img 
                  src={previewSrc[pdfPage]} 
                  alt="" 
                  style={viewerStyle} // 스타일 통일
                />
                {previewSrc.length > 1 && (
                  <div className="page-controls" style={{ position: 'absolute', bottom: '20px', left: '50%', transform: 'translateX(-50%)', background: 'rgba(0,0,0,0.5)', padding: '5px 15px', borderRadius: '20px', color: '#fff' }}>
                    <button onClick={() => setPdfPage(p => Math.max(0, p - 1))} style={{background:'none', border:'none', color:'white', cursor:'pointer'}}>◀</button>
                    <span style={{margin: '0 10px'}}>{pdfPage + 1} / {previewSrc.length}</span>
                    <button onClick={() => setPdfPage(p => Math.min(previewSrc.length - 1, p + 1))} style={{background:'none', border:'none', color:'white', cursor:'pointer'}}>▶</button>
                  </div>
                )}
              </>
            )}

            {/* 2. 동영상 파일 모드 */}
            {mediaType === 'video' && (
              <video 
                src={previewSrc} 
                controls 
                style={viewerStyle} // 스타일 통일
              />
            )}

            {/* 3. 유튜브 모드 */}
            {mediaType === 'youtube' && (
              <iframe
                width="100%"
                height="100%"
                src={`https://www.youtube.com/embed/${previewSrc}`}
                title="YouTube video player"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                style={{ backgroundColor: '#000' }}
              ></iframe>
            )}
          </div>

          <div className="camera-panel">
            {cameraOn && <img src={`${API}/stream?t=${streamTs}`} alt="" />}
            <div className="camera-buttons">
              <button onClick={runVLM}>VLM 분석</button>
              <button onClick={stopCamera}>종료</button>
            </div>
          </div>
        </div>
      )}

      {/* ================= MODAL ================= */}
      {modalType && (
        <div className="modal-overlay" onClick={() => setModalType(null)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <h2>
              {modalType === "guide" && "AI 코치 가이드라인"}
              {modalType === "usage" && "AI 코치 사용 방법"}
              {modalType === "setting" && "설정"}
            </h2>
            {modalType === "guide" && (
              <ul>
                <li>카메라에 상반신이 명확히 보이도록 한다.</li>
                <li>매뉴얼과 동일한 순서로 동작한다.</li>
              </ul>
            )}
            {modalType === "usage" && (
              <ol>
                <li>매뉴얼(영상/이미지/링크) 업로드</li>
                <li>코칭 시작 및 동작 수행</li>
                <li>VLM 분석</li>
              </ol>
            )}
            {modalType === "setting" && (
              <>
                <label>
                  <input type="checkbox" checked={confirmExit} onChange={e => setConfirmExit(e.target.checked)}/> 종료 확인
                </label>
                <label>
                  <input type="checkbox" checked={showAlert} onChange={e => setShowAlert(e.target.checked)}/> 분석 알림
                </label>
              </>
            )}
            <button onClick={() => setModalType(null)}>닫기</button>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;