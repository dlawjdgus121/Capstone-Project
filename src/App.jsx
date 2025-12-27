import React, { useState } from 'react';
import CardSwap, { Card } from './CardSwap';
import './App.css'; // 대소문자 수정 (app.css -> App.css)
import { BsCodeSlash, BsSliders, BsCircleFill, BsCloudUpload, BsFileEarmarkPdf, BsFileEarmarkImage } from 'react-icons/bs';

// 이미지 경로 수정 (src/img 폴더 기준)
// 파일명 오타(Realiable) 그대로 유지하여 import
import reliableImg from './img/image_Realiable.png';
import smoothImg from './img/image_Smooth.png';
import customizableImg from './img/image_Customizable.png';

function App() {
  const [isStarted, setIsStarted] = useState(false);
  const [isActive, setIsActive] = useState(false); // 드래그 상태
  const [uploadedFile, setUploadedFile] = useState(null); // 업로드된 파일

  // 유효성 검사 (JPG, PDF만 허용)
  const validateFile = (file) => {
    // MIME 타입 확인 (jpeg, pdf)
    const validTypes = ['application/pdf', 'image/jpeg', 'image/jpg'];
    if (validTypes.includes(file.type)) {
      setUploadedFile(file);
      console.log("파일 선택됨:", file.name);
      // 추후 서버 전송 로직 추가 위치
    } else {
      alert("JPG 또는 PDF 파일만 업로드 가능합니다.");
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) validateFile(file);
  };

  // 드래그 이벤트 핸들러
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsActive(true);
  };

  const handleDragLeave = () => {
    setIsActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsActive(false);
    const file = e.dataTransfer.files[0];
    if (file) validateFile(file);
  };

  return (
    <div className="app-screen">
      
      {/* 1. 시작 전 (랜딩 페이지) */}
      {!isStarted && (
        <>
          <div className="text-section fade-in">
            <h1>AI 코치 서비스에<br/>오신 걸 환영합니다</h1>
            <button className="start-button" onClick={() => setIsStarted(true)}>
              시작하기
            </button>
          </div>

          <div className="card-swap-wrapper fade-in">
            <CardSwap width={650} height={850} cardDistance={50} verticalDistance={120}>
              <Card>
                <div className="card-title-group">
                  <BsCodeSlash className="card-icon reliable-icon" />
                  <span>Reliable</span>
                </div>
                <p>언제나 정확한 답변</p>
                <img src={reliableImg} alt="Reliable" className="card-image" />
              </Card>
              <Card>
                <div className="card-title-group">
                  <BsCircleFill className="card-icon smooth-icon" />
                  <span>Smooth</span>
                </div>
                <p>직관적인 학습 경험</p>
                <img src={smoothImg} alt="Smooth" className="card-image" />
              </Card>
              <Card>
                <div className="card-title-group">
                  <BsSliders className="card-icon customizable-icon" />
                  <span>Customizable</span>
                </div>
                <p>맞춤형 학습 경로</p>
                <img src={customizableImg} alt="Customizable" className="card-image" />
              </Card>
            </CardSwap>
          </div>
        </>
      )}

      {/* 2. 시작 후 (업로드 페이지 - 중앙 정렬 & 드래그) */}
      {isStarted && (
        <div className="upload-container fade-in">
          <div className="glass-window">
            <h2>파일 업로드</h2>
            <p className="upload-desc">JPG 또는 PDF 파일을 이곳에 드래그하세요</p>
            
            <label 
              className={`upload-area ${isActive ? 'active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input 
                type="file" 
                accept=".jpg, .jpeg, .pdf"
                onChange={handleFileChange}
                className="file-input"
              />
              <div className="upload-content">
                {uploadedFile ? (
                  <div className="file-preview">
                    {uploadedFile.type.includes('pdf') ? 
                      <BsFileEarmarkPdf size={60}/> : 
                      <BsFileEarmarkImage size={60}/>
                    }
                    <span className="file-name">{uploadedFile.name}</span>
                    <span className="re-upload-text">파일 변경하기</span>
                  </div>
                ) : (
                  <>
                    <BsCloudUpload size={60} className="upload-icon"/>
                    <span className="main-text">드래그 앤 드롭</span>
                    <span className="sub-text">또는 클릭하여 파일 선택</span>
                    <div className="support-tags">
                      <span className="tag">PDF</span>
                      <span className="tag">JPG</span>
                    </div>
                  </>
                )}
              </div>
            </label>

            <button className="back-button" onClick={() => {
              setIsStarted(false);
              setUploadedFile(null);
            }}>
              뒤로 가기
            </button>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;