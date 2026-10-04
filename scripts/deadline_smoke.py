"""时限冒烟：deadline_ms 受限请求及时退出（HTTP 504），随后普通请求仍可完成。

可直接运行（不依赖服务，走内存 TestClient）；verify 流程在服务健康后通过
BASE_URL 走 HTTP。用法::

    python scripts/deadline_smoke.py            # 内存 TestClient
    BASE_URL=http://web:8000 python scripts/deadline_smoke.py   # 走 HTTP
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smoke import build_case, check_result, expected_payload  # noqa: E402

# 504 响应不得混入的字段：参数、分配、目标值与无解结论
FORBIDDEN_KEYS = ("solvable", "reason", "parameters", "objective", "assignments", "discarded")


def _post_via_http(base_url, payload):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/wafer-grids/reconstruct",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:  # 4xx/5xx 也是有效响应，读取正文
        return e.code, json.loads(e.read())


def _poster():
    base_url = os.environ.get("BASE_URL")
    if base_url:
        print(f"==> 通过 HTTP ({base_url}) 时限冒烟")
        return lambda payload: _post_via_http(base_url, payload)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    print("==> 内存 TestClient 时限冒烟")

    def post(payload):
        r = client.post("/api/wafer-grids/reconstruct", json=payload)
        return r.status_code, r.json()

    return post


def main():
    points, _ = build_case()
    post = _poster()

    # 1) 1ms 时限：完整枚举不可能完成，须及时返回 504 且不混入任何结论
    limited = expected_payload(points)
    limited["deadline_ms"] = 1
    t0 = time.monotonic()
    status, body = post(limited)
    elapsed_limited = time.monotonic() - t0
    assert status == 504, (status, body)
    assert body["status"] == "deadline_exceeded", body
    assert body["deadline_ms"] == 1, body
    assert body["retryable"] is True, body
    for key in FORBIDDEN_KEYS:
        assert key not in body, f"504 响应不得包含 {key}: {body}"
    print(f"  受限请求 {elapsed_limited * 1000:.1f} ms 内返回 504（deadline_ms=1）")

    # 2) 随后普通请求仍可完成：服务未被受限请求占住，漏读/杂点冒烟不回归
    t0 = time.monotonic()
    status, body = post(expected_payload(points))
    elapsed_normal = time.monotonic() - t0
    assert status == 200, (status, body)
    check_result(body)
    print(f"  普通请求 {elapsed_normal * 1000:.1f} ms 完成裁决")

    # 3) 受限请求确实及时退出（远早于一次完整裁决）
    assert elapsed_limited < max(1.0, elapsed_normal / 2), (
        f"受限请求耗时 {elapsed_limited:.3f}s，未明显快于完整裁决 {elapsed_normal:.3f}s"
    )
    print("==> 时限冒烟通过：受限请求及时退出，服务未被占住，普通请求照常完成")


if __name__ == "__main__":
    main()
