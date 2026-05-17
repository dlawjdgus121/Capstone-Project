## HTTP/3 실행

기존 `python main.py`는 `uvicorn`으로 HTTP/1.1 서버를 실행합니다.
HTTP/3로 테스트하려면 TLS 인증서가 필요하므로 아래 순서로 실행하세요.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_dev_cert.py
.\.venv\Scripts\python.exe run_http3.py
```

브라우저 주소는 `https://localhost:8443`입니다.

HTTP/3는 QUIC을 쓰기 때문에 방화벽/공유기에서 TCP뿐 아니라 UDP `8443`도 열려 있어야 합니다.
휴대폰에서 접속하려면 자체 서명 인증서를 신뢰시키거나, 실제 도메인과 정상 TLS 인증서를 사용해야 합니다.
