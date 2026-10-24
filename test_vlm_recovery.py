"""서버 STEP_CACHE가 비었을 때 클라이언트가 자동 복구하는지 확인.

실제 추론 서버(RUNPOD_INFERENCE_URL)에 붙어서 돈다. 서버가 없으면 skip.
실행: python test_vlm_recovery.py
"""
import asyncio
import glob
import os

from config import BASE_DIR, RUNPOD_INFERENCE_BASE_URL
import vlm


def pick_manual_url() -> str:
    hits = glob.glob(os.path.join(BASE_DIR, "outputs", "*", "*.jpg"))
    assert hits, "테스트용 매뉴얼 이미지가 outputs/ 에 없음"
    return "/" + os.path.relpath(hits[0], BASE_DIR).replace("\\", "/")


async def main() -> None:
    if not RUNPOD_INFERENCE_BASE_URL:
        print("SKIP: RUNPOD_INFERENCE_URL 미설정")
        return

    manual_url = pick_manual_url()
    desc = "recovery self-check"
    step_id = vlm.make_vlm_step_id(manual_url, desc)
    camera = open(os.path.join(BASE_DIR, manual_url.lstrip("/")), "rb").read()

    # 1) sync: 서버가 모르는 id는 등록 집합에서 떨어져야 한다
    vlm.REGISTERED_STEP_IDS.add("ghost_id_server_never_saw")
    vlm.WARMED_STEP_IDS.add("ghost_id_server_never_saw")
    await vlm.sync_registered_steps_from_server()
    assert "ghost_id_server_never_saw" not in vlm.REGISTERED_STEP_IDS, "sync가 유령 id를 안 버렸다"
    assert "ghost_id_server_never_saw" not in vlm.WARMED_STEP_IDS
    print("OK 1/2  sync가 서버에 없는 step_id를 버린다")

    # 2) 복구: 서버는 모르는데 로컬은 등록됐다고 믿는 상태를 만든다
    #    -> set_step을 건너뛴 predict가 400을 맞고, 재등록 후 성공해야 한다
    vlm.REGISTERED_STEP_IDS.add(step_id)
    vlm.last_vlm_step_key = step_id
    result = await vlm.call_runpod_inference(manual_url, camera, desc)
    assert result.get("result") != "ERROR", f"복구 실패: {result}"
    assert step_id in vlm.REGISTERED_STEP_IDS, "복구 후 재등록이 안 됐다"
    print(f"OK 2/2  캐시 미스에서 자동 복구 — result={result.get('result')}")

    await vlm.close_runpod_http_client()


if __name__ == "__main__":
    asyncio.run(main())
