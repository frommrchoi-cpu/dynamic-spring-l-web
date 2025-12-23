# dynamic_spring_l.py
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna()


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _format_poly_equation(coefs: np.ndarray, var: str = "β") -> str:
    deg = len(coefs) - 1
    terms = []
    for i, c in enumerate(coefs):
        p = deg - i
        if abs(c) < 1e-12:
            continue
        c_str = f"{c:.10g}"
        if p == 0:
            terms.append(f"{c_str}")
        elif p == 1:
            terms.append(f"{c_str}·{var}")
        else:
            terms.append(f"{c_str}·{var}^{p}")
    if not terms:
        return "aL = 0"
    return ("aL = " + " + ".join(terms)).replace("+ -", "- ")


@dataclass
class RegressionResult:
    degree: int
    r2: float
    coefs: np.ndarray
    equation: str


# -----------------------------
# Core model
# -----------------------------
class ALInterpolator:
    """
    1) 각 S/2Ro 곡선에 대해 β-aL을 PCHIP 보간
    2) 입력 S/2r0에 대해 인접한 두 S레벨 사이 선형 보간
    """

    def __init__(self, calibration_csv_path: str):
        self.calibration_csv_path = calibration_csv_path
        self.raw_df = self._load_calibration(calibration_csv_path)
        self.curves = self._build_curves(self.raw_df)  # S -> (B, aL, PCHIP)
        self.s_levels = sorted(self.curves.keys())

        if len(self.s_levels) < 2:
            raise ValueError(f"S/2Ro 곡선이 최소 2개 필요합니다. 현재: {self.s_levels}")

        all_b = np.concatenate([self.curves[s][0] for s in self.s_levels])
        self.beta_min = float(np.min(all_b))
        self.beta_max = float(np.max(all_b))

    @staticmethod
    def _load_calibration(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        rename = {}
        for c in df.columns:
            k = c.strip().lower()
            if k in ["s/2ro", "s/2r0", "s_2ro", "s_2r0", "s_over_2r0", "s_over_2ro"]:
                rename[c] = "S/2Ro"
            elif k in ["b", "beta", "β"]:
                rename[c] = "B"
            elif k in ["al", "a_l", "al".lower()]:
                rename[c] = "aL"
        if rename:
            df = df.rename(columns=rename)

        required = {"S/2Ro", "B", "aL"}
        if not required.issubset(df.columns):
            raise ValueError(f"보정 CSV에 {required} 열이 필요합니다. 현재: {list(df.columns)}")

        return _to_numeric_df(df[["S/2Ro", "B", "aL"]])

    @staticmethod
    def _build_curves(df: pd.DataFrame) -> Dict[float, Tuple[np.ndarray, np.ndarray, PchipInterpolator]]:
        curves = {}
        for s, g in df.groupby("S/2Ro"):
            g = g.sort_values("B")
            b = g["B"].to_numpy(float)
            al = g["aL"].to_numpy(float)

            _, idx = np.unique(b, return_index=True)
            b, al = b[idx], al[idx]

            if len(b) < 4:
                raise ValueError(f"S/2Ro={s} 점이 부족합니다(>=4 필요). 현재 {len(b)}")

            curves[float(s)] = (b, al, PchipInterpolator(b, al, extrapolate=True))
        return curves

    def _bracket_s(self, s: float) -> Tuple[float, float]:
        L = self.s_levels
        if s <= L[0]:
            return L[0], L[1]
        if s >= L[-1]:
            return L[-2], L[-1]
        for i in range(len(L) - 1):
            if L[i] <= s <= L[i + 1]:
                return L[i], L[i + 1]
        return L[-2], L[-1]

    def aL_at(self, s_over_2r0: float, beta: float) -> float:
        s = float(s_over_2r0)
        b = float(beta)
        s1, s2 = self._bracket_s(s)

        _, _, f1 = self.curves[s1]
        _, _, f2 = self.curves[s2]
        al1 = float(f1(b))
        al2 = float(f2(b))

        if abs(s2 - s1) < 1e-12:
            return al1
        w = (s - s1) / (s2 - s1)
        return (1 - w) * al1 + w * al2

    def curve_at(self, s_over_2r0: float, betas: np.ndarray) -> np.ndarray:
        betas = np.asarray(betas, float)
        return np.array([self.aL_at(s_over_2r0, bb) for bb in betas], float)

    def curve_at_existing_s(self, s_level: float, betas: np.ndarray) -> np.ndarray:
        """Return curve only using that S-level's PCHIP (no S-direction interpolation)."""
        s_level = float(s_level)
        if s_level not in self.curves:
            raise ValueError(f"S/2r0={s_level} 곡선이 CSV에 없습니다. 현재 레벨: {self.s_levels}")
        _, _, f = self.curves[s_level]
        betas = np.asarray(betas, float)
        return np.array([float(f(bb)) for bb in betas], float)

    def regression_for_curve(
        self,
        s_over_2r0: float,
        beta_grid: np.ndarray,
        target_r2: float = 0.99,
        max_degree: int = 12
    ) -> RegressionResult:
        x = np.asarray(beta_grid, float)
        y = self.curve_at(s_over_2r0, x)

        best = None
        for deg in range(2, max_degree + 1):
            coefs = np.polyfit(x, y, deg)
            yhat = np.polyval(coefs, x)
            r2 = _r2_score(y, yhat)
            res = RegressionResult(
                degree=deg, r2=float(r2), coefs=coefs,
                equation=_format_poly_equation(coefs, var="β")
            )
            if best is None or res.r2 > best.r2:
                best = res
            if res.r2 >= target_r2:
                return res
        return best

    def plot_with_reference_curves(
        self,
        s_input: float,
        beta_input: float,
        ref_s_levels: List[float] = None,
        target_r2: float = 0.99
    ) -> Tuple[float, RegressionResult, plt.Figure]:
        """
        Always plot:
        - Reference curves at S=2 and S=5 (if present)
        - Interpolated curve at input S
        - Mark the (beta_input, aL) point on the interpolated curve
        """
        if ref_s_levels is None:
            ref_s_levels = [2.0, 5.0]

        betas = np.linspace(self.beta_min, self.beta_max, 400)

        fig = plt.figure()

        # 1) plot reference curves (S=2,5)
        for s_ref in ref_s_levels:
            if float(s_ref) in self.curves:
                y_ref = self.curve_at_existing_s(float(s_ref), betas)
                plt.plot(betas, y_ref, label=f"Base curve (S/2r0={float(s_ref):g})")
            else:
                # if not in CSV, just skip (no crash)
                pass

        # 2) plot interpolated curve at input S
        y_int = self.curve_at(s_input, betas)
        plt.plot(betas, y_int, label=f"Interpolated curve (S/2r0={float(s_input):g})")

        # 3) mark the input point
        aL_point = self.aL_at(s_input, beta_input)
        plt.scatter([beta_input], [aL_point], label=f"Input point β={beta_input:g}, aL={aL_point:.6g}")

        # styling
        plt.xlabel("β")
        plt.ylabel("aL")
        plt.title("Lateral Displacement Interaction Factor (aL)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # regression for interpolated curve
        reg = self.regression_for_curve(s_input, betas, target_r2=target_r2)
        plt.suptitle(f"Regression@S={float(s_input):g}: deg={reg.degree}, R²={reg.r2:.6f}",
                     y=0.98, fontsize=10)

        return aL_point, reg, fig


# -----------------------------
# CLI loop (always show plot)
# -----------------------------
def run_cli(calib: str):
    model = ALInterpolator(calib)

    print("Dynamic Spring L - CLI (그래프 항상 표시)")
    print(f"Calibration: {calib}")
    print(f"S/2r0 levels: {model.s_levels}")
    print(f"β range: {model.beta_min:.3g} ~ {model.beta_max:.3g}")
    print("quit: q\n")

    while True:
        s_in = input("S/2r0 > ").strip()
        if s_in.lower() in ["q", "quit", "exit"]:
            break
        b_in = input("β(B) > ").strip()
        if b_in.lower() in ["q", "quit", "exit"]:
            break

        try:
            s = float(s_in)
            b = float(b_in)
        except ValueError:
            print("숫자로 입력해주세요.\n")
            continue

        aL, reg, fig = model.plot_with_reference_curves(s, b, ref_s_levels=[2.0, 5.0], target_r2=0.99)

        print(f"\naL = {aL:.10f}")
        print(f"Regression deg={reg.degree}, R²={reg.r2:.6f}")
        print(reg.equation)
        plt.show()
        print("")


# -----------------------------
# Web app (Streamlit) - always show plot after calculate
# -----------------------------
def run_web(calib: str):
    import streamlit as st

    st.set_page_config(page_title="Dynamic Spring L", layout="wide")
    st.title("Dynamic Spring L — aL 보간/그래프/회귀식 (기본곡선 2,5 + 입력 보간곡선)")

    model = ALInterpolator(calib)

    with st.sidebar:
        st.header("보정 데이터 정보")
        st.write(f"- 파일: `{os.path.basename(calib)}`")
        st.write(f"- S/2r0 레벨: {model.s_levels}")
        st.write(f"- β 범위(데이터 기반): {model.beta_min:.3g} ~ {model.beta_max:.3g}")
        st.caption("※ 그래프: S=2, S=5 기본곡선 + 입력 S 보간곡선")

    col1, col2 = st.columns([1, 2])

    with col1:
        s_val = st.number_input("S/2r₀", value=2.0, step=0.1, format="%.4f")
        b_val = st.number_input("β (B)", value=30.0, step=1.0, format="%.4f")
        target_r2 = st.slider("회귀식 목표 R²", min_value=0.90, max_value=0.999, value=0.99, step=0.001)
        run_btn = st.button("계산")

    if run_btn:
        aL, reg, fig = model.plot_with_reference_curves(
            s_val, b_val, ref_s_levels=[2.0, 5.0], target_r2=float(target_r2)
        )

        with col1:
            st.subheader("결과")
            st.write(f"- aL = **{aL:.10f}**")
            st.write(f"- 회귀식 차수: **{reg.degree}**")
            st.write(f"- R²: **{reg.r2:.6f}**")
            st.text("회귀식:")
            st.code(reg.equation)

        with col2:
            st.subheader("그래프")
            st.pyplot(fig, clear_figure=True)


# -----------------------------
# Entrypoint (subcommand has its own --calib)
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_cli = sub.add_parser("cli", help="CLI 반복 입력 (그래프 항상)")
    p_cli.add_argument("--calib", default="aL_poulos.csv", help="Calibration CSV path")

    p_web = sub.add_parser("web", help="웹앱 실행")
    p_web.add_argument("--calib", default="aL_poulos.csv", help="Calibration CSV path")

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(args.calib)
    elif args.mode == "web":
        run_web(args.calib)


if __name__ == "__main__":
    main()
