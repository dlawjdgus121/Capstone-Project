import asyncio
import json

from livekit import rtc

from config import LIVEKIT_TOKEN, LIVEKIT_URL
from state import state
from vision import process_video_track


async def run_livekit():
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("📹 LiveKit 비디오 트랙 수신 시작")
            asyncio.create_task(process_video_track(track))

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        try:
            payload = json.loads(data.data.decode("utf-8"))
            if payload.get("type") == "ping":
                state["client_send_time"] = payload.get("send_time")
                resp = json.dumps({"type": "pong", "client_time": payload.get("send_time")})
                asyncio.create_task(room.local_participant.publish_data(resp.encode("utf-8")))
            elif payload.get("type") == "metrics":
                state["last_rtt"] = float(payload.get("rtt", 0.0))
        except Exception:
            pass

    try:
        if not LIVEKIT_TOKEN:
            print("🚨 [ERROR] LIVEKIT_TOKEN이 .env 파일에 없습니다!")
            return
        await room.connect(LIVEKIT_URL, LIVEKIT_TOKEN)
        print("✅ [SYSTEM] LiveKit 서버 접속 성공")
    except Exception as e:
        print(f"🚨 [ERROR] LiveKit 접속 실패: {e}")
