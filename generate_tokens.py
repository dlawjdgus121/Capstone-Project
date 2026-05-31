import os
from dotenv import load_dotenv

load_dotenv()

try:
    from livekit import api
except ImportError:
    print("❌ pip install python-dotenv livekit livekit-api 먼저 실행하세요")
    exit(1)

API_KEY    = os.getenv("LIVEKIT_API_KEY", "")
API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
ROOM       = os.getenv("LIVEKIT_ROOM", "cd")

if not API_KEY or not API_SECRET:
    print("❌ .env에 LIVEKIT_API_KEY, LIVEKIT_API_SECRET을 설정하세요")
    exit(1)

def make_token(identity: str, can_publish: bool, can_subscribe: bool) -> str:
    token = api.AccessToken(api_key=API_KEY, api_secret=API_SECRET)
    token.with_identity(identity).with_name(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=ROOM,
        can_publish=can_publish,
        can_subscribe=can_subscribe,
        can_publish_data=True,
    ))
    return token.to_jwt()

server_token = make_token("server", can_publish=False, can_subscribe=True)
mobile_token = make_token("mb",     can_publish=True,  can_subscribe=False)
pc_token     = make_token("pc",     can_publish=False, can_subscribe=True)

print("\n✅ 토큰 생성 완료 — 아래를 .env에 붙여넣으세요\n")
print(f"LIVEKIT_TOKEN={server_token}")
print(f"MOBILE_TOKEN={mobile_token}")
print(f"PC_TOKEN={pc_token}")
print()