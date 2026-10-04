"""HTTP 层测试。"""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402

client = TestClient(app)

VB = {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}}


def payload(points, **over):
    base = {
        "points": [{"id": i, "x": x, "y": y} for i, x, y in points],
        "rows": 3,
        "cols": 3,
        "max_outliers": 0,
        "tolerance": 0,
        "origin_bounds": VB,
        "row_vector_bounds": VB,
        "col_vector_bounds": VB,
    }
    base.update(over)
    return base


def exact_points():
    return [
        (r * 3 + c + 1, 2 * r, 2 * c)
        for r in range(3)
        for c in range(3)
    ]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_reconstruct_ok():
    r = client.post("/api/wafer-grids/reconstruct", json=payload(exact_points()))
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is True
    assert body["parameters"]["origin"] == [0, 0]
    assert body["parameters"]["row_vector"] == [2, 0]
    assert body["parameters"]["col_vector"] == [0, 2]
    assert len({(a["row"], a["col"]) for a in body["assignments"]}) == 9
    for a in body["assignments"]:
        assert a["predicted"] == [a["x"], a["y"]]


def test_reconstruct_no_solution_body():
    pts = [(i, 100 + 3 * i, 200 + 3 * i) for i in range(1, 8)]
    r = client.post("/api/wafer-grids/reconstruct", json=payload(pts, max_outliers=2))
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is False
    assert body["reason"]


def test_duplicate_ids_rejected():
    pts = [(1, 0, 0)] * 7
    r = client.post("/api/wafer-grids/reconstruct", json=payload(pts))
    assert r.status_code == 422


def test_interval_span_rejected():
    p = payload(exact_points())
    p["origin_bounds"] = {"x": {"lo": 0, "hi": 7}, "y": {"lo": 0, "hi": 0}}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert "跨度" in r.text


def test_counts_out_of_range():
    p = payload(exact_points())
    p["max_outliers"] = 3
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422
    p = payload(exact_points())
    p["points"] = p["points"][:6]
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422
    p = payload(exact_points())
    p["rows"] = 8
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422


def test_deadline_ms_range_validated():
    p = payload(exact_points())
    p["deadline_ms"] = 0
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422
    p = payload(exact_points())
    p["deadline_ms"] = 5001
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422
    p = payload(exact_points())
    p["deadline_ms"] = 5000
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 200


def test_deadline_exceeded_returns_504_without_conclusion():
    # 1ms 时限：完整枚举（约数百毫秒）不可能完成，须返回 504
    p = payload(exact_points(), deadline_ms=1)
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 504
    body = r.json()
    assert body["status"] == "deadline_exceeded"
    assert body["deadline_ms"] == 1
    assert body["retryable"] is True
    # 不得混入参数、分配或无解结论
    for key in (
        "solvable",
        "reason",
        "parameters",
        "objective",
        "assignments",
        "discarded",
    ):
        assert key not in body, f"504 响应不得包含 {key}"


def test_normal_request_completes_after_deadline_exceeded():
    # 受限请求 504 后服务不被占住：随后的普通请求照常完成裁决
    limited = payload(exact_points(), deadline_ms=1)
    assert client.post("/api/wafer-grids/reconstruct", json=limited).status_code == 504
    r = client.post("/api/wafer-grids/reconstruct", json=payload(exact_points()))
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is True
    assert body["parameters"]["origin"] == [0, 0]


def test_generous_deadline_matches_unlimited_response():
    # 时限内完成时响应须与无时限请求完全一致（未经降级）
    r_limited = client.post(
        "/api/wafer-grids/reconstruct", json=payload(exact_points(), deadline_ms=5000)
    )
    r_plain = client.post("/api/wafer-grids/reconstruct", json=payload(exact_points()))
    assert r_limited.status_code == r_plain.status_code == 200
    assert r_limited.json() == r_plain.json()


def test_omitted_deadline_keeps_unsolvable_behavior():
    pts = [(i, 100 + 3 * i, 200 + 3 * i) for i in range(1, 8)]
    r = client.post("/api/wafer-grids/reconstruct", json=payload(pts, max_outliers=2))
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is False
    assert "status" not in body  # 几何无解与时限未裁决是两种不同响应
