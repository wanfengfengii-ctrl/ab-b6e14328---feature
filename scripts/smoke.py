"""复原冒烟：含漏读标记与划痕亮点（杂点）的栅格复原。

可直接运行（不依赖服务）；verify 流程在服务健康后通过 BASE_URL 走 HTTP，
同时保留对核心算法的直测。用法::

    python scripts/smoke.py            # 直测求解器
    BASE_URL=http://web:8000 python scripts/smoke.py   # 走 HTTP
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver import reconstruct  # noqa: E402

BOUNDS = {
    "origin": ([-3, 3], [-3, 3]),
    "row_vector": ([-3, 3], [-3, 3]),
    "col_vector": ([-3, 3], [-3, 3]),
}


def build_case():
    """4x4 栅格，O=(0,0)，行向量 A=(3,0)，列向量 B=(0,3)，det=9。

    故意漏读 3 个格位，并混入 2 个划痕亮点；再给 3 个标记施加分量 ≤1 的扰动。
    """
    true_cells = {
        (r, c): (3 * r, 3 * c) for r in range(4) for c in range(4)
    }
    missing = {(1, 1), (2, 3), (3, 0), (0, 3)}  # 漏读 4 格
    # 抖动均为曼哈顿 1：任何其他基向量若最大残差同为 1，也必须在众多
    # 精确点上付出更大残差和，真栅格凭第三级目标（残差总和）唯一胜出
    jitter = {(0, 2): (1, 0), (2, 1): (0, -1), (3, 3): (-1, 0)}
    points = []
    pid = 1
    for r in range(4):
        for c in range(4):
            if (r, c) in missing:
                continue
            x, y = true_cells[(r, c)]
            if (r, c) in jitter:
                dx, dy = jitter[(r, c)]
                x += dx
                y += dy
            points.append((pid, x, y))
            pid += 1
    # 两个划痕亮点（杂点），故意打散顺序
    points.extend(
        [
            (90, 17, -5),
            (91, -8, 14),
        ]
    )
    # 打乱坐标顺序，验证“无序坐标恢复”
    import random

    random.Random(42).shuffle(points)
    return points, missing


def expected_payload(points):
    return {
        "points": [{"id": i, "x": x, "y": y} for i, x, y in points],
        "rows": 4,
        "cols": 4,
        "max_outliers": 2,
        "tolerance": 1,
        "origin_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
        "row_vector_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
        "col_vector_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
    }


# 14 个散乱点：容差 3、全宽区间下完整枚举约 2.7s（几何无解），
# 使 100ms 时限必然在裁决途中到期，而 5000ms 足以完成裁决。
SLOW_POINTS = [
    (1, -18, -34), (2, -2, 10), (3, -5, 26), (4, -16, -24),
    (5, 20, -22), (6, 10, 36), (7, 6, -14), (8, -13, -28),
    (9, -9, -18), (10, -6, 10), (11, -7, -16), (12, -21, -9),
    (13, -31, -22), (14, 13, 14),
]


def slow_payload(deadline_ms=None):
    p = {
        "points": [{"id": i, "x": x, "y": y} for i, x, y in SLOW_POINTS],
        "rows": 4,
        "cols": 4,
        "max_outliers": 2,
        "tolerance": 3,
        "origin_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
        "row_vector_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
        "col_vector_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
    }
    if deadline_ms is not None:
        p["deadline_ms"] = deadline_ms
    return p


def check_result(result):
    assert result["solvable"] is True, result.get("reason")
    p = result["parameters"]
    assert p["origin"] == [0, 0], p
    assert p["row_vector"] == [3, 0], p
    assert p["col_vector"] == [0, 3], p
    assert p["determinant"] == 9

    obj = result["objective"]
    assert obj["discarded_count"] == 2, obj
    assert obj["max_manhattan_residual"] == 1, obj
    assert obj["total_manhattan_residual"] == 3, obj  # 三个抖动点各 1

    adopted = [a for a in result["assignments"] if a["adopted"]]
    discarded = [a for a in result["assignments"] if not a["adopted"]]
    assert len(adopted) == 12
    assert len(discarded) == 2
    assert {a["id"] for a in discarded} == {90, 91}

    # 每个被采用标记落回真实格位
    true_xy_to_rc = {(3 * r, 3 * c): (r, c) for r in range(4) for c in range(4)}
    cells = set()
    for a in adopted:
        assert max(abs(a["residual"][0]), abs(a["residual"][1])) <= 1
        pr, pc = true_xy_to_rc[tuple(a["predicted"])]
        assert (a["row"], a["col"]) == (pr, pc)
        cells.add((a["row"], a["col"]))
    assert len(cells) == len(adopted), "两个标记占用了同一格位"

    # 弃点证据
    for d in result["discarded"]:
        assert d["id"] in (90, 91)
        assert d["nearest_cell"] is not None
        assert d["nearest_inf_residual"] > 1
        assert d["cells_within_tolerance"] == []
    print(
        f"  弃点 {[d['id'] for d in result['discarded']]}，"
        f"目标 = (k={obj['discarded_count']}, "
        f"max={obj['max_manhattan_residual']}, sum={obj['total_manhattan_residual']})"
    )


def http_post_json(base_url, payload, timeout=30):
    """POST JSON，返回 (status_code, body_dict)，不把 4xx/5xx 当作异常。"""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/wafer-grids/reconstruct",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check_deadline_over_http(base_url):
    """受限请求及时 504 退出，随后放宽时限的普通请求仍能完成裁决。

    且 504 响应体只表达“本次尚未裁决”，不混入参数、分配或无解结论。
    """
    import time

    t0 = time.monotonic()
    status, body = http_post_json(base_url, slow_payload(deadline_ms=100))
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert status == 504, (status, body)
    assert elapsed_ms < 1500, f"受限请求未及时退出：{elapsed_ms:.0f}ms"
    assert body == {
        "status": "deadline_exceeded",
        "deadline_ms": 100,
        "retryable": True,
    }, body
    assert not (
        {"parameters", "assignments", "discarded", "solvable", "reason"} & set(body)
    )
    print(f"==> 受限请求在 {elapsed_ms:.0f}ms 内以 504 及时退出")

    # 随后放宽时限：同一请求完成裁决，明确给出几何无解结论
    status2, body2 = http_post_json(base_url, slow_payload(deadline_ms=5000))
    assert status2 == 200, (status2, body2)
    assert body2["solvable"] is False and body2["reason"], body2
    print("==> 放宽时限后的普通请求完成裁决（几何无解，非尚未裁决）")


def main():
    points, _ = build_case()
    base_url = os.environ.get("BASE_URL")
    if base_url:
        status, result = http_post_json(base_url, expected_payload(points))
        assert status == 200, (status, result)
        print(f"==> 通过 HTTP ({base_url}) 冒烟")
        check_deadline_over_http(base_url)
    else:
        result = reconstruct(points, 4, 4, 1, 2, BOUNDS)
        print("==> 直测求解器冒烟（时限链路需经 HTTP 由 verify 覆盖）")
    check_result(result)
    print("==> 冒烟通过：漏读 4 格 + 2 划痕亮点均正确处理")


if __name__ == "__main__":
    main()
