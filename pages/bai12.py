"""pages/bai12.py — Bài 12: Đồ án tích hợp AIDEOM-VN, bản 1 file.

Mục tiêu của file này:
- Tích hợp đủ 6 module AIDEOM-VN trong Mục 14 của đề vào một file duy nhất để dùng trực tiếp trong thư mục pages/.
- Đọc đủ 3 CSV: vietnam_macro_2020_2025.csv, vietnam_regions_2024.csv, vietnam_sectors_2024.csv.
- Có 5 kịch bản chính sách S1-S5, trong đó S5 được tính tự động bằng mô hình thỏa hiệp đơn giản.
- Dashboard Streamlit có 4 tab: Tổng quan, Phân bổ, Kịch bản so sánh, Cảnh báo rủi ro.

Lưu ý học thuật:
Đề kỹ thuật gốc yêu cầu M1-M5 tách thành các file .py độc lập và có pytest. Vì người dùng muốn một file duy nhất,
file này gom các module thành các nhóm hàm trong cùng một file. Khi giảng viên chấm đúng tiêu chí phần mềm nghiêm
ngặt, bản tách module vẫn tốt hơn. Khi cần chạy nhanh trong app hiện tại, bản một file này phù hợp hơn.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from utils import bai_header, end_padding, info_box, section_title
except Exception:
    def bai_header(so, ten, mo_ta, cap_do, tools, thoi_luong):
        st.title(f"Bài {so}. {ten}")
        st.caption(mo_ta)
        st.write(f"**Cấp độ:** {cap_do} · **Tools:** {', '.join(tools)} · **Thời lượng:** {thoi_luong}")

    def section_title(title, icon=""):
        st.markdown(f"### {icon} {title}")

    def info_box(html, bg="#F3F6F4", border="#1A6B3C", icon="💡"):
        st.markdown(
            f"<div style='background:{bg};border-left:4px solid {border};padding:0.9rem 1rem;"
            f"border-radius:10px;margin:0.65rem 0;'><b>{icon}</b> {html}</div>",
            unsafe_allow_html=True,
        )

    def end_padding():
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"
C2 = "#2E8B57"
C3 = "#4CAF72"
CBLUE = "#1976D2"
CRED = "#E53935"
CORANGE = "#E65100"
CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình chung
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_K = 0.33
BETA_L = 0.42
GAMMA_D = 0.10
DELTA_AI = 0.08
THETA_H = 0.07

K_HIST = np.array([16500, 17800, 19600, 21300, 23500, 25900.0])
L_HIST = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
D_HIST = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
AI_HIST = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
H_HIST = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

REGION_SHORT = {
    "Northern Midlands and Mountains": "TDMNPB",
    "Red River Delta": "ĐBSH",
    "North Central and South Central Coast": "BTB+DHMT",
    "Central Highlands": "Tây Nguyên",
    "Southeast": "ĐNB",
    "Mekong Delta": "ĐBSCL",
}

SCENARIO_FIXED = {
    "S1_Truyen_thong": (
        "S1. Truyền thống",
        np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
        "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu.",
    ),
    "S2_So_hoa_nhanh": (
        "S2. Số hóa nhanh",
        np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
        "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số.",
    ),
    "S3_AI_dan_dat": (
        "S3. AI dẫn dắt",
        np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
        "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu.",
    ),
    "S4_Bao_trum_so": (
        "S4. Bao trùm số",
        np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
        "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số.",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def _find_csv(name: str) -> Path:
    """Tìm CSV theo nhiều vị trí thường gặp khi chạy Streamlit.

    Hỗ trợ cả tên có hậu tố do ChatGPT tạo khi upload như vietnam_macro_2020_2025(1).csv hoặc (2).csv.
    """
    stem = name.replace(".csv", "")
    candidates: List[Path] = []
    for base in [Path("data"), Path("."), Path("..") / "data", Path("/mnt/data")]:
        candidates.extend([
            base / name,
            base / f"{stem}(1).csv",
            base / f"{stem}(2).csv",
            base / f"{stem}(3).csv",
        ])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Không tìm thấy {name}. Hãy đặt file trong thư mục data/ hoặc cùng thư mục app.py."
    )


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc đủ 3 CSV của đề: macro, sectors, regions."""
    macro = pd.read_csv(_find_csv("vietnam_macro_2020_2025.csv"))
    sectors = pd.read_csv(_find_csv("vietnam_sectors_2024.csv"))
    regions = pd.read_csv(_find_csv("vietnam_regions_2024.csv"))
    return macro, sectors, regions


# ══════════════════════════════════════════════════════════════════════════════
# M1 — Dự báo kinh tế Macro 2026–2030
# ══════════════════════════════════════════════════════════════════════════════
def validate_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra nhanh số GDP 2024–2025 trong CSV với số tham chiếu trong đề/NSO."""
    ref = {2024: 11511.9, 2025: 12847.6}
    rows = []
    for year, ref_val in ref.items():
        found = macro.loc[macro["year"] == year, "GDP_trillion_VND"]
        if found.empty:
            rows.append({"Năm": year, "GDP CSV": np.nan, "GDP tham chiếu": ref_val, "Sai lệch": np.nan, "Kết luận": "THIẾU"})
            continue
        csv_val = float(found.iloc[0])
        rows.append({
            "Năm": year,
            "GDP CSV": csv_val,
            "GDP tham chiếu": ref_val,
            "Sai lệch": abs(csv_val - ref_val),
            "Kết luận": "OK" if abs(csv_val - ref_val) < 1e-6 else "CHECK",
        })
    return pd.DataFrame(rows)


def calibrate_tfp(macro: pd.DataFrame) -> pd.DataFrame:
    """Hiệu chỉnh A_t trong Cobb-Douglas mở rộng dựa trên macro 2020–2025."""
    ordered = macro.sort_values("year")
    y = ordered["GDP_trillion_VND"].to_numpy(float)
    core = (K_HIST ** ALPHA_K) * (L_HIST ** BETA_L) * (D_HIST ** GAMMA_D) * (AI_HIST ** DELTA_AI) * (H_HIST ** THETA_H)
    A = y / core
    return pd.DataFrame({"year": ordered["year"].to_numpy(int), "GDP": y, "TFP_A": A})


def production(A: float, K: float, L: float, D: float, AI: float, H: float) -> float:
    """Hàm sản xuất Cobb-Douglas mở rộng."""
    return float(A * (K ** ALPHA_K) * (L ** BETA_L) * (max(D, 1e-9) ** GAMMA_D) * (max(AI, 1e-9) ** DELTA_AI) * (max(H, 1e-9) ** THETA_H))


def simulate_path(alloc: np.ndarray, macro: pd.DataFrame, start: int = 2026, end: int = 2030, budget: float = 1800.0) -> pd.DataFrame:
    """Mô phỏng quỹ đạo 2026–2030 cho một vector phân bổ K/D/AI/H.

    budget đơn vị nghìn tỷ VND/năm. Vì D, AI, H không cùng đơn vị với K, dùng hệ số chuyển đổi etaD/etaAI/etaH.
    """
    tfp0 = float(calibrate_tfp(macro)["TFP_A"].iloc[-1]) * 1.012
    K, L, D, AI, H = 27500.0, 53.9, 20.3, 86.0, 30.0
    tfp = tfp0
    etaD, etaAI, etaH = 0.0016, 0.0055, 0.0009
    rows = []
    for year in range(start, end + 1):
        Y = production(tfp, K, L, D, AI, H)
        rows.append({"year": year, "GDP": Y, "K": K, "D": D, "AI": AI, "H": H, "TFP": tfp})
        IK, ID, IAI, IH = alloc * budget
        K = 0.95 * K + IK
        D = min(40.0, 0.88 * D + etaD * ID)
        AI = min(155.0, 0.85 * AI + etaAI * IAI)
        H = min(48.0, 0.98 * H + etaH * IH)
        L *= 1.004
        tfp *= 1 + 0.012 + 0.0010 * (D / 20) + 0.0008 * (AI / 90) + 0.0012 * (H / 30)
    out = pd.DataFrame(rows)
    out["GDP_growth_pct"] = out["GDP"].pct_change() * 100
    return out


# ══════════════════════════════════════════════════════════════════════════════
# M2 — Đánh giá sẵn sàng số bằng TOPSIS + Entropy
# ══════════════════════════════════════════════════════════════════════════════
def entropy_weights(X: np.ndarray) -> np.ndarray:
    Xp = np.maximum(np.asarray(X, dtype=float), 1e-12)
    P = Xp / Xp.sum(axis=0)
    E = -np.sum(P * np.log(P + 1e-12), axis=0) / np.log(Xp.shape[0])
    d = np.maximum(1 - E, 0)
    if d.sum() <= 1e-12:
        return np.ones(Xp.shape[1]) / Xp.shape[1]
    return d / d.sum()


def topsis(X: np.ndarray, w: np.ndarray, is_benefit: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1e-12
    R = X / denom
    V = R * w
    ideal = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    anti = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    sp = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    sn = np.sqrt(((V - anti) ** 2).sum(axis=1))
    return sn / (sp + sn + 1e-12)


def rank_regions(regions: pd.DataFrame, method: str = "expert") -> pd.DataFrame:
    """Xếp hạng 6 vùng theo TOPSIS. method='expert' hoặc 'entropy'."""
    criteria = [
        "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD",
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
        "gini_coef",
    ]
    X = regions[criteria].to_numpy(float)
    is_benefit = np.array([True, True, True, True, True, True, True, False])
    if method == "entropy":
        w = entropy_weights(X)
    else:
        w = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    score = topsis(X, w, is_benefit)
    out = regions[["region_id", "region_name_en", "digital_index_0_100", "ai_readiness_0_100"]].copy()
    out["Vùng"] = out["region_name_en"].map(REGION_SHORT).fillna(out["region_name_en"])
    out["TOPSIS"] = score
    out["Rank"] = out["TOPSIS"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("Rank").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# M3 — Tối ưu/phân bổ ngân sách ngành-vùng-thời gian
# ══════════════════════════════════════════════════════════════════════════════
def c5_check(regions: pd.DataFrame) -> Dict[str, float | bool]:
    """Kiểm tra phát hiện từ Bài 4/Bài 7: C5 gốc λ=0.70 có khả thi hay không."""
    d0 = regions.sort_values("region_id")["digital_index_0_100"].to_numpy(float)
    min_M = float(d0.max())
    threshold = 0.70 * min_M
    weakest_max = float(d0.min() + 0.002 * 12000)
    return {
        "M tối thiểu": min_M,
        "Ngưỡng λM": threshold,
        "Vùng yếu tối đa": weakest_max,
        "Khả thi?": weakest_max >= threshold,
        "λ tối đa khả thi": weakest_max / min_M,
    }


def allocation_proxy(regions: pd.DataFrame, budget: float = 50000.0) -> pd.DataFrame:
    """Proxy phân bổ vùng × hạng mục dựa trên readiness, giữ sàn vùng yếu.

    Đây là phiên bản tích hợp gọn của Bài 4 + Bài 8 để phục vụ dashboard Bài 12.
    """
    ranked = rank_regions(regions, method="expert").sort_values("region_id").reset_index(drop=True)
    score = ranked["TOPSIS"].to_numpy(float)
    base = np.ones(6) * 5000.0
    extra = max(0.0, budget - base.sum())
    weight = score / score.sum()
    total = base + extra * weight
    total = np.minimum(total, 12000.0)
    rows = []
    for i, row in ranked.iterrows():
        if row["TOPSIS"] > 0.70:
            alloc = np.array([0.10, 0.30, 0.45, 0.15])
        elif row["TOPSIS"] > 0.35:
            alloc = np.array([0.20, 0.35, 0.25, 0.20])
        else:
            alloc = np.array([0.30, 0.25, 0.10, 0.35])
        vals = total[i] * alloc
        rows.append({"Vùng": row["Vùng"], "I": vals[0], "D": vals[1], "AI": vals[2], "H": vals[3], "Tổng": vals.sum()})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# M4 — Mô phỏng lao động AI/automation/đào tạo lại
# ══════════════════════════════════════════════════════════════════════════════
def _selected_labor_sectors(sectors: pd.DataFrame) -> pd.DataFrame:
    # Bài 9 bỏ Mining và Healthcare, giữ 8 ngành lớn.
    ids = [1, 2, 3, 5, 6, 7, 8, 9]
    return sectors[sectors["sector_id"].isin(ids)].sort_values("sector_id").reset_index(drop=True).copy()


def labor_balanced(sectors: pd.DataFrame, budget: float = 30000.0, ai_share_override: float | None = None, h_share_override: float | None = None) -> pd.DataFrame:
    """Mô phỏng NetJob theo ngành.

    Nếu ai_share_override/h_share_override không None, dùng để đánh giá một kịch bản chính sách cụ thể.
    """
    df = _selected_labor_sectors(sectors)
    A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    B1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
    C1c = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    D1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])
    risk = df["automation_risk_pct"].to_numpy(float) / 100.0
    labor_weight = df["labor_million"].to_numpy(float)
    budget_i = budget * labor_weight / labor_weight.sum()
    net_ai_per = A1 - C1c * risk

    if ai_share_override is None or h_share_override is None:
        xAI = np.where(net_ai_per > B1, budget_i * 0.60, budget_i * 0.15)
        xH = budget_i - xAI
    else:
        total_ai_h = max(ai_share_override + h_share_override, 1e-12)
        ai_ratio = ai_share_override / total_ai_h
        xAI = budget_i * ai_ratio
        xH = budget_i - xAI

    # Đảm bảo Displaced <= RetrainingCapacity bằng cách chuyển bớt AI sang H nếu cần.
    displaced = C1c * risk * xAI
    min_xH = displaced / D1
    need = min_xH > xH
    xH[need] = min_xH[need]
    xAI[need] = np.maximum(0.0, budget_i[need] - xH[need])

    new_job = A1 * xAI
    displaced = C1c * risk * xAI
    upgrade = B1 * xH
    net = new_job + upgrade - displaced

    out = df[["sector_name_en", "labor_million", "automation_risk_pct"]].copy()
    out["x_AI"] = xAI
    out["x_H"] = xH
    out["NewJob"] = new_job
    out["DisplacedJob"] = displaced
    out["UpgradeJob"] = upgrade
    out["NetJob"] = net
    return out


def labor_score_for_allocation(alloc: np.ndarray, sectors: pd.DataFrame) -> float:
    lab = labor_balanced(sectors, ai_share_override=float(alloc[2]), h_share_override=float(alloc[3]))
    return float(lab["NetJob"].sum())


# ══════════════════════════════════════════════════════════════════════════════
# M5 — Đánh giá rủi ro Cyber, môi trường, phụ thuộc và SP/robust proxy
# ══════════════════════════════════════════════════════════════════════════════
def risk_from_allocation(alloc: np.ndarray) -> Dict[str, float]:
    K, D, AI, H = [float(v) for v in alloc]
    cyber = max(0.0, 100.0 * (0.50 * AI + 0.25 * D - 0.30 * H))
    env = max(0.0, 100.0 * (0.45 * K + 0.35 * AI - 0.15 * D))
    dep = max(0.0, 100.0 * (0.55 * AI + 0.20 * D - 0.25 * H))
    resilience = 100.0 - min(100.0, 0.35 * cyber + 0.35 * env + 0.30 * dep)
    return {"Cyber Risk": cyber, "Environment Risk": env, "Dependency Risk": dep, "Resilience": resilience}


def risk_metrics(scenarios: Dict[str, Tuple[str, np.ndarray, str]], summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (name, alloc, desc) in scenarios.items():
        metrics = risk_from_allocation(alloc)
        gdp2030 = float(summary.loc[summary["Mã"] == key, "GDP 2030"].iloc[0])
        rows.append({"Mã": key, "Kịch bản": name, **metrics, "GDP 2030": gdp2030})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Tạo kịch bản S5 từ mô hình AIDEOM-VN
# ══════════════════════════════════════════════════════════════════════════════
def _grid_allocations(step: float = 0.05) -> List[np.ndarray]:
    n = int(round(1 / step))
    arrs = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                alloc = np.array([a, b, c, d], dtype=float) * step
                # Giữ một mức tối thiểu thực tế để kịch bản không cực đoan.
                if np.all(alloc >= 0.05):
                    arrs.append(alloc)
    return arrs


def derive_s5_allocation(macro: pd.DataFrame, sectors: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Tính S5 bằng tìm kiếm lưới trên vector phân bổ K/D/AI/H.

    Hàm điểm thỏa hiệp gồm: tăng trưởng GDP 2030, NetJob, rủi ro thấp, và cân bằng phân bổ.
    """
    candidates = _grid_allocations(0.05)
    rows = []
    for alloc in candidates:
        path = simulate_path(alloc, macro)
        gdp2030 = float(path["GDP"].iloc[-1])
        netjob = labor_score_for_allocation(alloc, sectors)
        risk = risk_from_allocation(alloc)
        risk_bad = risk["Cyber Risk"] + risk["Environment Risk"] + risk["Dependency Risk"]
        balance = 1.0 - float(np.std(alloc) / (np.mean(alloc) + 1e-12))
        rows.append({
            "K": alloc[0], "D": alloc[1], "AI": alloc[2], "H": alloc[3],
            "GDP2030": gdp2030, "NetJob": netjob, "RiskBad": risk_bad, "Balance": balance,
        })
    df = pd.DataFrame(rows)
    # Chuẩn hóa mọi trục về 0–1; GDP/NetJob/Balance là benefit, RiskBad là cost.
    def norm_col(col: str, benefit: bool = True) -> pd.Series:
        vals = df[col].astype(float)
        rng = float(np.ptp(vals.to_numpy())) + 1e-12
        z = (vals - vals.min()) / rng
        return z if benefit else 1 - z
    df["Score"] = (
        0.40 * norm_col("GDP2030", True)
        + 0.25 * norm_col("NetJob", True)
        + 0.20 * norm_col("RiskBad", False)
        + 0.15 * norm_col("Balance", True)
    )
    best = df.sort_values("Score", ascending=False).iloc[0]
    alloc = best[["K", "D", "AI", "H"]].to_numpy(float)
    return alloc, df.sort_values("Score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_scenarios(macro: pd.DataFrame, sectors: pd.DataFrame) -> Tuple[Dict[str, Tuple[str, np.ndarray, str]], pd.DataFrame]:
    s5_alloc, grid = derive_s5_allocation(macro, sectors)
    scenarios = dict(SCENARIO_FIXED)
    scenarios["S5_Toi_uu_can_bang"] = (
        "S5. Tối ưu cân bằng",
        s5_alloc,
        "Kết quả mô hình AIDEOM-VN: tối ưu thỏa hiệp giữa GDP, NetJob, rủi ro và cân bằng phân bổ.",
    )
    return scenarios, grid


def scenario_summary(macro: pd.DataFrame, scenarios: Dict[str, Tuple[str, np.ndarray, str]]) -> pd.DataFrame:
    rows = []
    for key, (name, alloc, desc) in scenarios.items():
        path = simulate_path(alloc, macro)
        first, last = path.iloc[0], path.iloc[-1]
        cagr = (last["GDP"] / first["GDP"]) ** (1 / (len(path) - 1)) - 1
        rows.append({
            "Mã": key,
            "Kịch bản": name,
            "Mô tả": desc,
            "K": alloc[0],
            "D": alloc[1],
            "AI": alloc[2],
            "H": alloc[3],
            "GDP 2030": last["GDP"],
            "CAGR %": cagr * 100,
            "D 2030": last["D"],
            "AI 2030": last["AI"],
            "H 2030": last["H"],
            "TFP 2030": last["TFP"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Kiểm tra nội bộ thay cho pytest khi làm bản 1 file
# ══════════════════════════════════════════════════════════════════════════════
def internal_checks(macro: pd.DataFrame, sectors: pd.DataFrame, regions: pd.DataFrame, scenarios: Dict[str, Tuple[str, np.ndarray, str]]) -> pd.DataFrame:
    checks = []
    checks.append(("Đọc macro CSV", len(macro) >= 6 and "GDP_trillion_VND" in macro.columns))
    checks.append(("Đọc sectors CSV", len(sectors) >= 10 and "automation_risk_pct" in sectors.columns))
    checks.append(("Đọc regions CSV", len(regions) >= 6 and "digital_index_0_100" in regions.columns))
    checks.append(("Có đủ 5 kịch bản S1-S5", len(scenarios) == 5))
    checks.append(("Mỗi kịch bản có tổng phân bổ = 100%", all(abs(v[1].sum() - 1.0) < 1e-9 for v in scenarios.values())))
    checks.append(("M1 calibrate TFP không lỗi", calibrate_tfp(macro)["TFP_A"].notna().all()))
    checks.append(("M2 TOPSIS trả đủ 6 vùng", len(rank_regions(regions)) == 6))
    checks.append(("M3 allocation có 6 vùng", len(allocation_proxy(regions)) == 6))
    checks.append(("M4 NetJob không âm tổng", labor_balanced(sectors)["NetJob"].sum() >= 0))
    checks.append(("M5 risk có đủ chỉ số", all(k in risk_from_allocation(np.array([0.25, 0.25, 0.25, 0.25])) for k in ["Cyber Risk", "Environment Risk", "Dependency Risk"])))
    return pd.DataFrame([{"Hạng mục kiểm tra": k, "Kết quả": "PASS" if ok else "FAIL"} for k, ok in checks])


# ══════════════════════════════════════════════════════════════════════════════
# M6 — Dashboard ra quyết định
# ══════════════════════════════════════════════════════════════════════════════
def render():
    bai_header(
        so="12",
        ten="Đồ án tích hợp — Nguyên mẫu AIDEOM-VN",
        mo_ta="Bản 1 file: tích hợp M1-M6, đọc 3 CSV, so sánh 5 kịch bản, cảnh báo rủi ro và khuyến nghị chính sách",
        cap_do="KHÓ",
        tools=["pandas", "numpy", "plotly", "streamlit"],
        thoi_luong="4–6 tuần",
    )

    macro, sectors, regions = load_data()
    scenarios, s5_grid = get_scenarios(macro, sectors)
    summary = scenario_summary(macro, scenarios)
    readiness_expert = rank_regions(regions, method="expert")
    readiness_entropy = rank_regions(regions, method="entropy")
    alloc = allocation_proxy(regions)
    labor = labor_balanced(sectors)
    risk = risk_metrics(scenarios, summary)
    c5 = c5_check(regions)

    info_box(
        "AIDEOM-VN gồm 6 module: <b>M1 Macro</b>, <b>M2 Readiness</b>, <b>M3 Allocation</b>, "
        "<b>M4 Labor</b>, <b>M5 Risk</b>, <b>M6 Dashboard</b>. Đây là bản <b>một file py</b> để dùng trực tiếp trong app.",
        bg="#E8F5E9",
        border=C1,
        icon="📐",
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan / M6", "Phân bổ / M2-M4", "Kịch bản so sánh / M1", "Cảnh báo rủi ro / M5"])

    with tab1:
        section_title("Tổng quan hệ thống AIDEOM-VN", "🧭")
        st.markdown("**Bản đồ 6 module trong file một `.py`:**")
        module_map = pd.DataFrame([
            {"Module": "M1", "Tên": "Dự báo kinh tế Macro", "Hàm chính trong file": "calibrate_tfp(), simulate_path(), scenario_summary()", "Đầu ra": "GDP, TFP, K/D/AI/H 2026–2030"},
            {"Module": "M2", "Tên": "Đánh giá sẵn sàng số", "Hàm chính trong file": "rank_regions(), topsis(), entropy_weights()", "Đầu ra": "TOPSIS, AI Readiness theo vùng"},
            {"Module": "M3", "Tên": "Tối ưu/phân bổ", "Hàm chính trong file": "allocation_proxy(), c5_check()", "Đầu ra": "Phân bổ vùng × I/D/AI/H"},
            {"Module": "M4", "Tên": "Mô phỏng lao động", "Hàm chính trong file": "labor_balanced()", "Đầu ra": "NetJob, DisplacedJob, UpgradeJob"},
            {"Module": "M5", "Tên": "Đánh giá rủi ro", "Hàm chính trong file": "risk_from_allocation(), risk_metrics()", "Đầu ra": "Cyber, Environment, Dependency, Resilience"},
            {"Module": "M6", "Tên": "Dashboard ra quyết định", "Hàm chính trong file": "render()", "Đầu ra": "4 tab trực quan và khuyến nghị"},
        ])
        st.dataframe(module_map, use_container_width=True, hide_index=True)
        best = summary.sort_values("GDP 2030", ascending=False).iloc[0]
        best_resilience = risk.sort_values("Resilience", ascending=False).iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("GDP 2030 cao nhất", best["Kịch bản"], f"{best['GDP 2030']:,.0f} ng.tỷ")
        k2.metric("Vùng readiness #1", readiness_expert.iloc[0]["Vùng"], f"TOPSIS {readiness_expert.iloc[0]['TOPSIS']:.3f}")
        k3.metric("NetJob cân bằng", f"{labor['NetJob'].sum():,.0f}", "việc làm ròng")
        k4.metric("Resilience tốt nhất", best_resilience["Kịch bản"], f"{best_resilience['Resilience']:.1f}/100")


        st.markdown("**Kiểm tra GDP 2024–2025 với số tham chiếu trong đề/NSO:**")
        st.dataframe(validate_macro(macro), use_container_width=True, hide_index=True)

        st.markdown("**5 kịch bản chính sách, trong đó S5 được mô hình tự tính:**")
        scen_table = pd.DataFrame([
            {"Mã": k, "Kịch bản": v[0], "K": v[1][0], "D": v[1][1], "AI": v[1][2], "H": v[1][3], "Mô tả": v[2]}
            for k, v in scenarios.items()
        ])
        st.dataframe(scen_table, use_container_width=True, hide_index=True)

        fig = px.bar(summary, x="Kịch bản", y="GDP 2030", color="CAGR %", text_auto=".0f", title="GDP 2030 theo 5 kịch bản")
        fig.update_layout(height=360, font=CF)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Xem top 10 phân bổ ứng viên dùng để chọn S5"):
            st.dataframe(s5_grid.head(10).round(4), use_container_width=True, hide_index=True)

    with tab2:
        section_title("M2–M4: Readiness, phân bổ và lao động", "🧩")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**M2 — TOPSIS xếp hạng 6 vùng, trọng số chuyên gia**")
            st.dataframe(readiness_expert[["Vùng", "digital_index_0_100", "ai_readiness_0_100", "TOPSIS", "Rank"]].round(4), use_container_width=True, hide_index=True)
            fig = px.bar(readiness_expert.sort_values("TOPSIS"), x="TOPSIS", y="Vùng", orientation="h", title="AI Readiness theo TOPSIS chuyên gia")
            fig.update_layout(height=330, font=CF)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.markdown("**M2 — So sánh TOPSIS chuyên gia và Entropy**")
            cmp = readiness_expert[["region_id", "Vùng", "Rank"]].rename(columns={"Rank": "Rank chuyên gia"}).merge(
                readiness_entropy[["region_id", "TOPSIS", "Rank"]].rename(columns={"TOPSIS": "TOPSIS entropy", "Rank": "Rank entropy"}),
                on="region_id",
                how="left",
            )
            st.dataframe(cmp[["Vùng", "Rank chuyên gia", "Rank entropy", "TOPSIS entropy"]].round(4), use_container_width=True, hide_index=True)

            st.markdown("**M3 — Proxy phân bổ vùng × hạng mục**")
            st.dataframe(alloc.round(0), use_container_width=True, hide_index=True)

        heat = alloc.set_index("Vùng")[["I", "D", "AI", "H"]]
        fig_h = px.imshow(heat, text_auto=".0f", aspect="auto", title="Heatmap phân bổ ngân sách vùng × hạng mục")
        fig_h.update_layout(height=360, font=CF)
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

        st.markdown("**M4 — NetJob cân bằng theo tỷ lệ lao động ngành**")
        st.dataframe(labor[["sector_name_en", "labor_million", "automation_risk_pct", "x_AI", "x_H", "NewJob", "DisplacedJob", "UpgradeJob", "NetJob"]].round(0), use_container_width=True, hide_index=True)
        fig_l = px.bar(labor.sort_values("NetJob"), x="NetJob", y="sector_name_en", orientation="h", title="NetJob ròng theo ngành")
        fig_l.update_layout(height=420, font=CF)
        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        section_title("M1 — Dự báo kinh tế Macro 2026–2030", "📈")
        info_box("Module 1 nằm ở nhóm hàm <b>calibrate_tfp()</b>, <b>production()</b>, <b>simulate_path()</b> và <b>scenario_summary()</b>. Đây là phần dùng dữ liệu macro 2020–2025 để hiệu chỉnh TFP và mô phỏng GDP theo 5 kịch bản.", bg="#E8F5E9", border=C1, icon="M1")
        chosen = st.multiselect("Chọn kịch bản", list(scenarios.keys()), default=list(scenarios.keys()))
        paths = []
        for key in chosen:
            path = simulate_path(scenarios[key][1], macro)
            path["Kịch bản"] = scenarios[key][0]
            paths.append(path)
        if paths:
            dfp = pd.concat(paths, ignore_index=True)
            fig1 = px.line(dfp, x="year", y="GDP", color="Kịch bản", markers=True, title="Quỹ đạo GDP")
            fig1.update_layout(height=360, font=CF)
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

            c1, c2 = st.columns(2)
            with c1:
                fig2 = px.line(dfp, x="year", y="AI", color="Kịch bản", markers=True, title="Quỹ đạo năng lực AI")
                fig2.update_layout(height=330, font=CF)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            with c2:
                fig3 = px.line(dfp, x="year", y="H", color="Kịch bản", markers=True, title="Quỹ đạo vốn nhân lực số")
                fig3.update_layout(height=330, font=CF)
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        st.markdown("**Bảng tổng hợp kết quả 2030 — đáp ứng yêu cầu test S1, S3, S5 và đủ 5 kịch bản:**")
        st.dataframe(summary.round(3), use_container_width=True, hide_index=True)

        st.markdown("**So sánh riêng 3 kịch bản bắt buộc S1, S3, S5:**")
        required = summary[summary["Mã"].isin(["S1_Truyen_thong", "S3_AI_dan_dat", "S5_Toi_uu_can_bang"])]
        st.dataframe(required.round(3), use_container_width=True, hide_index=True)

    with tab4:
        section_title("M5 — Cảnh báo rủi ro và khuyến nghị", "🚨")
        st.dataframe(risk.round(2), use_container_width=True, hide_index=True)
        fig_r = go.Figure()
        for col in ["Cyber Risk", "Environment Risk", "Dependency Risk"]:
            fig_r.add_trace(go.Bar(name=col, x=risk["Kịch bản"], y=risk[col]))
        fig_r.update_layout(barmode="group", height=360, title="Rủi ro theo kịch bản", font=CF)
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

        section_title("Cảnh báo mô hình C5", "⚠️")
        st.dataframe(pd.DataFrame([c5]).round(4), use_container_width=True, hide_index=True)
        if not c5["Khả thi?"]:
            info_box(
                "Ràng buộc công bằng C5 gốc của Bài 4/Bài 7 với λ=0,70 không khả thi trong bộ tham số đề bài. "
                "Khi nộp, nên ghi rõ phát hiện này thay vì ép mô hình cho ra nghiệm sai.",
                bg="#FFF8E1",
                border=CORANGE,
                icon="⚠️",
            )

        section_title("Khuyến nghị chính sách", "✅")
        s5 = summary.loc[summary["Mã"] == "S5_Toi_uu_can_bang"].iloc[0]
        info_box(
            f"S5 là phương án thỏa hiệp do mô hình chạy ra: K={s5['K']:.2f}, D={s5['D']:.2f}, AI={s5['AI']:.2f}, H={s5['H']:.2f}. "
            "Phương án này cân bằng tăng trưởng GDP, NetJob, rủi ro và tránh cực đoan hóa chính sách.",
            bg="#E3F2FD",
            border=CBLUE,
            icon="✅",
        )
        info_box(
            "Nếu chọn S3 AI dẫn dắt, cần tăng đầu tư H và an ninh dữ liệu để giảm Cyber/Dependency Risk. "
            "Nếu chọn S4 bao trùm, NetJob và resilience tốt hơn nhưng cần chấp nhận tốc độ GDP 2030 thấp hơn.",
            bg="#E8F5E9",
            border=C1,
            icon="💡",
        )

    section_title("Kết luận đồ án", "🧾")
    st.markdown(
        "Bài 12 là nguyên mẫu tích hợp các mô hình từ Bài 1–11 vào một hệ hỗ trợ ra quyết định. "
        "Trong bản một file này, các module M1–M6 được tổ chức thành các nhóm hàm độc lập trong cùng một file để dễ đưa vào app hiện tại. "
        "Mô hình hỗ trợ so sánh kịch bản, phát hiện ràng buộc vô nghiệm và minh họa đánh đổi chính sách; "
        "nó không thay thế quyết định chính trị - xã hội của con người."
    )
    end_padding()


if __name__ == "__main__":
    render()
