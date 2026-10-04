"""FastAPI 应用：晶圆标记栅格复原服务。"""

import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from .solver import DeadlineExceeded, reconstruct

app = FastAPI(
    title="Wafer Grid Reconstruction API",
    description="从带编号的无序整数坐标中恢复栅格原点、基向量与格位分配。",
    version="1.0.0",
)


class Marker(BaseModel):
    id: int = Field(..., description="标记唯一编号")
    x: int
    y: int


class Interval(BaseModel):
    """单个分量的闭区间 [lo, hi]，跨度不超过 6。"""

    lo: int
    hi: int

    @model_validator(mode="after")
    def _check(self):
        if self.hi < self.lo:
            raise ValueError(f"闭区间下界 {self.lo} 不得大于上界 {self.hi}")
        if self.hi - self.lo > 6:
            raise ValueError(
                f"区间 [{self.lo}, {self.hi}] 跨度 {self.hi - self.lo} 超过 6"
            )
        return self


class VectorBounds(BaseModel):
    x: Interval
    y: Interval


class ReconstructRequest(BaseModel):
    points: list[Marker] = Field(..., min_length=7, max_length=14)
    rows: int = Field(..., ge=3, le=7)
    cols: int = Field(..., ge=3, le=7)
    max_outliers: int = Field(..., ge=0, le=2)
    tolerance: int = Field(..., ge=0, description="逐分量（L∞）坐标容差")
    origin_bounds: VectorBounds
    row_vector_bounds: VectorBounds
    col_vector_bounds: VectorBounds
    deadline_ms: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        description="单次复原的计算时限（毫秒）；省略时不设限，行为与既有版本一致",
    )

    @model_validator(mode="after")
    def _check_points(self):
        ids = [p.id for p in self.points]
        if len(set(ids)) != len(ids):
            raise ValueError("标记编号必须唯一")
        return self


def _bounds_pair(vb: VectorBounds):
    return ([vb.x.lo, vb.x.hi], [vb.y.lo, vb.y.hi])


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/wafer-grids/reconstruct")
async def reconstruct_grid(req: ReconstructRequest):
    # 按编号排序：字典序决胜项定义在“按编号排列的分配序列”上
    points = sorted(((p.id, p.x, p.y) for p in req.points), key=lambda t: t[0])
    bounds = {
        "origin": _bounds_pair(req.origin_bounds),
        "row_vector": _bounds_pair(req.row_vector_bounds),
        "col_vector": _bounds_pair(req.col_vector_bounds),
    }
    deadline = None
    if req.deadline_ms is not None:
        deadline = time.monotonic() + req.deadline_ms / 1000.0
    try:
        result = await run_in_threadpool(
            reconstruct,
            points,
            req.rows,
            req.cols,
            req.tolerance,
            req.max_outliers,
            bounds,
            deadline,
        )
    except DeadlineExceeded:
        # 时限到期：裁决尚未完成。只报告“本次未裁决”，不得混入任何参数、
        # 分配或无解结论；调用方可放宽 deadline_ms 后重试。
        return JSONResponse(
            status_code=504,
            content={
                "status": "deadline_exceeded",
                "deadline_ms": req.deadline_ms,
                "retryable": True,
                "detail": (
                    f"在 {req.deadline_ms} ms 时限内未能完成全局最优裁决；"
                    "这不代表几何上无解，可放宽 deadline_ms 后重试。"
                ),
            },
        )
    if not result.get("solvable"):
        # 几何上无解不是请求格式错误：以 200 返回明确的无解原因，
        # 由响应体 solvable=false 标识。
        return result
    return result
