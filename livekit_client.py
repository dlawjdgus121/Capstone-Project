import asyncio
import json

from livekit import rtc

from config import LIVEKIT_TOKEN, LIVEKIT_URL
from state import state
from vision import process_video_track


async def run_livekit():
    room = rtc.Room()
    video_task = None

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        nonlocal video_task
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("📹 LiveKit 비디오 트랙 수신 시작")
            video_task = asyncio.create_task(process_video_track(track))

    @room.on("disconnected")
    def on_disconnected():
        nonlocal video_task
        print("🔌 LiveKit 연결 끊김 — 비디오 처리 중단")
        if video_task and not video_task.done():
            video_task.cancel()
        state["log"] = "모바일 연결 끊김"

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
        # room 연결 유지
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        print("🔌 LiveKit 연결 취소됨 — 정리 중")
        if video_task and not video_task.done():
            video_task.cancel()
            try:
                await video_task
            except asyncio.CancelledError:
                pass
        await room.disconnect()
    except Exception as e:
        print(f"🚨 [ERROR] LiveKit 접속 실패: {e}")
