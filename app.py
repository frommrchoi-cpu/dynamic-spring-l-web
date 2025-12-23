# app.py
# -*- coding: utf-8 -*-
import io
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dynamic_spring_l_core import ALInterpolator

CALIB_PATH = os.getenv("CALIB_PATH", "aL_poulos.csv")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = ALInterpolator(CALIB_PATH)

def make_plot_png(s_val: float, b_val: float, target_r2: float) -> bytes:
    betas = np.linspace(model.beta_min, model.beta_max, 400)

    # base curves 2 & 5 (if exist)
    fig = plt.figure(figsize=(10, 6))
    for s_ref in [2.0, 5.0]:
        if float(s_ref) in model.curves:
            y_ref = model.curve_at_existing_s(float(s_ref), betas)
            plt.plot(betas, y_ref, label=f"Base curve (S/2r0={s_ref:g})")

    # interpolated curve at input S
    y_int = model.curve_at(s_val, betas)
    plt.plot(betas, y_int, label=f"Interpolated curve (S/2r0={s_val:g})")

    # point
    aL = model.aL_at(s_val, b_val)
    plt.scatter([b_val], [aL], label=f"Input β={b_val:g}, aL={aL:.6g}")

    reg = model.regression_for_curve(s_val, betas, target_r2=target_r2)

    plt.xlabel("β")
    plt.ylabel("aL")
    plt.title("Lateral Displacement Interaction Factor (aL)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.suptitle(f"Regression@S={s_val:g}: deg={reg.degree}, R²={reg.r2:.6f}", y=0.98, fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_s": 2.0,
            "default_b": 30.0,
            "default_r2": 0.99,
            "result": None,
        },
    )

@app.post("/", response_class=HTMLResponse)
def compute(
    request: Request,
    s_val: float = Form(...),
    b_val: float = Form(...),
    target_r2: float = Form(0.99),
):
    aL = model.aL_at(s_val, b_val)
    betas = np.linspace(model.beta_min, model.beta_max, 400)
    reg = model.regression_for_curve(s_val, betas, target_r2=target_r2)

    # plot image served via /plot?...
    ts = int(time.time() * 1000)
    plot_url = f"/plot?s={s_val}&b={b_val}&r2={target_r2}&t={ts}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_s": s_val,
            "default_b": b_val,
            "default_r2": target_r2,
            "result": {
                "aL": aL,
                "deg": reg.degree,
                "r2": reg.r2,
                "eq": reg.equation,
                "plot_url": plot_url,
            },
        },
    )

@app.get("/plot")
def plot(s: float, b: float, r2: float = 0.99, t: int = 0):
    png = make_plot_png(s, b, r2)
    return Response(content=png, media_type="image/png")
