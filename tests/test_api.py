"""HTTP 层测试。"""

import os
import sys
import time

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


# 14 个散乱点：容差 3、全宽区间下完整枚举约 2.7s（几何无解），
# 足以让 100ms 时限在裁决途中到期。
SLOW_POINTS = [
    (1, -18, -34), (2, -2, 10), (3, -5, 26), (4, -16, -24),
    (5, 20, -22), (6, 10, 36), (7, 6, -14), (8, -13, -28),
    (9, -9, -18), (10, -6, 10), (11, -7, -16), (12, -21, -9),
    (13, -31, -22), (14, 13, 14),
]

WIDE_VB = {"lo": -3, "hi": 3}


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


def test_deadline_ms_out_of_range():
    for bad in (0, -1, 5001, 10000):
        r = client.post(
            "/api/wafer-grids/reconstruct", json=payload(exact_points(), deadline_ms=bad)
        )
        assert r.status_code == 422, bad


def test_deadline_returns_504_without_partial_conclusion():
    p = payload(
        SLOW_POINTS,
        rows=4,
        cols=4,
        max_outliers=2,
        tolerance=3,
        deadline_ms=100,
    )
    # 三个分量区间全部放宽
    for key in ("origin_bounds", "row_vector_bounds", "col_vector_bounds"):
        p[key] = {"x": WIDE_VB, "y": WIDE_VB}
    t0 = time.monotonic()
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert r.status_code == 504
    assert elapsed_ms < 1500, elapsed_ms
    body = r.json()
    assert body == {
        "status": "deadline_exceeded",
        "deadline_ms": 100,
        "retryable": True,
    }
    # 不得混入参数、分配或无解结论
    assert "parameters" not in body
    assert "assignments" not in body
    assert "discarded" not in body
    assert "solvable" not in body
    assert "reason" not in body


def test_request_after_expired_deadline_still_completes():
    # 受限请求及时退出后，随后的普通（宽松时限）请求仍可完成并给出最优解
    slow = payload(
        SLOW_POINTS,
        rows=4,
        cols=4,
        max_outliers=2,
        tolerance=3,
        deadline_ms=100,
    )
    for key in ("origin_bounds", "row_vector_bounds", "col_vector_bounds"):
        slow[key] = {"x": WIDE_VB, "y": WIDE_VB}
    r1 = client.post("/api/wafer-grids/reconstruct", json=slow)
    assert r1.status_code == 504

    # 同一几何无解请求放宽到 5s：必须完成并明确区分“几何无解”与“尚未裁决”
    slow2 = dict(slow)
    slow2["deadline_ms"] = 5000
    r2 = client.post("/api/wafer-grids/reconstruct", json=slow2)
    assert r2.status_code == 200
    assert r2.json()["solvable"] is False
    assert r2.json()["reason"]

    # 后续普通可解请求不受影响
    r3 = client.post(
        "/api/wafer-grids/reconstruct", json=payload(exact_points())
    )
    assert r3.status_code == 200
    assert r3.json()["parameters"]["origin"] == [0, 0]


def test_completed_within_deadline_is_uncompromised_optimum():
    # 在时限内完成时仍返回未经降级的原最优结果
    r = client.post(
        "/api/wafer-grids/reconstruct",
        json=payload(exact_points(), deadline_ms=5000),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["parameters"]["origin"] == [0, 0]
    assert body["parameters"]["row_vector"] == [2, 0]
    assert body["parameters"]["col_vector"] == [0, 2]
    assert body["objective"]["discarded_count"] == 0


def test_omitted_deadline_response_shape_unchanged():
    r = client.post(
        "/api/wafer-grids/reconstruct", json=payload(exact_points())
    )
    body = r.json()
    assert set(body) == {
        "solvable", "objective", "parameters", "assignments", "discarded"
    }
