"""pages/bai9.py — Bài 9: Tác động AI tới thị trường lao động Việt Nam

Phiên bản đã kiểm tra lại:
- Đọc lao động và automation risk từ vietnam_sectors_2024.csv.
- Giải mô hình tuyến tính bằng CVXPY, fallback SciPy/HiGHS nếu thiếu CVXPY.
- Phân biệt rõ 2 ngưỡng ở câu 9.4.2:
  (i) NetJob >= 0; (ii) Displaced <= RetrainingCapacity.
- Giữ kết quả mô hình gốc đúng toán học nhưng giải thích vì sao nghiệm góc không đủ thực tế.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import bai_header, end_padding, info_box, section_title

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ── Style ───────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

# ── 8 ngành dùng trong Bài 9, bỏ Khai khoáng và Y tế ───────────────────────
SECTOR_ORDER_EN = [
    "Agriculture-Forestry-Fishery",
    "Manufacturing",
    "Construction",
    "Wholesale-Retail",
    "Finance-Banking-Insurance",
    "Logistics-Transport-Warehousing",
    "Information-Communication-IT",
    "Education-Training",
]
SECTORS_VI = [
    "Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"
]
SHORT = ["NLTS", "CN chế biến", "Xây dựng", "BB-BL", "Tài chính", "Logistics", "CNTT", "Giáo dục"]
N = 8
BUDGET = 30000.0

# Hệ số đề bài: việc làm/tỷ VND
A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5], dtype=float)   # NewJob từ AI
B1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0], dtype=float)  # Upgrade từ H
C1_DISP = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5], dtype=float)
D1_CAP = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0], dtype=float)

# Fallback đúng theo bảng đề
L_FALLBACK = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15], dtype=float)
RISK_FALLBACK = np.array([18, 42, 25, 38, 52, 35, 28, 22], dtype=float) / 100.0


def _find_csv(filename_stem: str) -> Path | None:
    """Tìm CSV trong thư mục hiện tại, data/, /mnt/data, hoặc biến môi trường."""
    candidates = []
    cwd = Path.cwd()
    for base in [cwd, cwd / "data", Path(__file__).resolve().parent if "__file__" in globals() else cwd,
                 Path("/mnt/data"), Path("/mnt/data/final_checked_bai1_7/data")]:
        candidates += [
            base / f"{filename_stem}.csv",
            base / f"{filename_stem}(1).csv",
        ]
    env_path = os.environ.get("BGP_DATA_DIR")
    if env_path:
        candidates += [Path(env_path) / f"{filename_stem}.csv", Path(env_path) / f"{filename_stem}(1).csv"]
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_sector_inputs():
    """Đọc labor và risk từ vietnam_sectors_2024.csv; fallback nếu thiếu file."""
    csv_path = _find_csv("vietnam_sectors_2024")
    if csv_path is None:
        return L_FALLBACK, RISK_FALLBACK, "fallback bảng đề"

    df = pd.read_csv(csv_path)
    required = {"sector_name_en", "labor_million", "automation_risk_pct"}
    if not required.issubset(df.columns):
        return L_FALLBACK, RISK_FALLBACK, f"fallback vì CSV thiếu cột: {csv_path.name}"

    sub = df.set_index("sector_name_en").reindex(SECTOR_ORDER_EN)
    if sub[["labor_million", "automation_risk_pct"]].isna().any().any():
        return L_FALLBACK, RISK_FALLBACK, f"fallback vì CSV thiếu ngành: {csv_path.name}"

    L = sub["labor_million"].to_numpy(dtype=float)
    risk = sub["automation_risk_pct"].to_numpy(dtype=float) / 100.0
    return L, risk, csv_path.name


def _solve_with_scipy(L, risk, cap5=False):
    """Giải LP bằng scipy.optimize.linprog; biến x=[xAI_1..xAI_8, xH_1..xH_8]."""
    if not HAS_SCIPY:
        return None
    net_ai = A1 - C1_DISP * risk
    coef = np.r_[net_ai, B1]
    c = -coef
    A_ub = []
    b_ub = []
    # Ngân sách
    A_ub.append(np.ones(2 * N)); b_ub.append(BUDGET)
    # NetJob_i >= 0 => -net_i <= 0
    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = -net_ai[i]
        row[N + i] = -B1[i]
        A_ub.append(row); b_ub.append(0.0)
    # Displaced_i <= RetrainingCapacity_i
    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = C1_DISP[i] * risk[i]
        row[N + i] = -D1_CAP[i]
        A_ub.append(row); b_ub.append(0.0)
    # Mở rộng 9.4.4: không ngành nào mất quá 5% lao động
    if cap5:
        for i in range(N):
            row = np.zeros(2 * N)
            row[i] = C1_DISP[i] * risk[i]
            A_ub.append(row); b_ub.append(0.05 * L[i] * 1_000_000)

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * (2 * N), method="highs")
    if not res.success:
        return {"status": res.message, "success": False}
    xAI = res.x[:N]
    xH = res.x[N:]
    return {"status": "optimal", "success": True, "xAI": xAI, "xH": xH, "objective": -res.fun}


def _solve_with_cvxpy(L, risk, cap5=False):
    if not HAS_CVXPY:
        return None
    xAI = cp.Variable(N, nonneg=True)
    xH = cp.Variable(N, nonneg=True)
    new = cp.multiply(A1, xAI)
    displaced = cp.multiply(C1_DISP * risk, xAI)
    upgrade = cp.multiply(B1, xH)
    net = new + upgrade - displaced
    cons = [
        cp.sum(xAI + xH) <= BUDGET,
        net >= 0,
        displaced <= cp.multiply(D1_CAP, xH),
    ]
    if cap5:
        cons.append(displaced <= 0.05 * L * 1_000_000)
    prob = cp.Problem(cp.Maximize(cp.sum(net)), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        try:
            prob.solve(solver=cp.SCS)
        except Exception:
            return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "success": False}
    return {"status": prob.status, "success": True, "xAI": np.array(xAI.value).ravel(),
            "xH": np.array(xH.value).ravel(), "objective": float(prob.value)}


@st.cache_data(show_spinner=False)
def solve_global(cap5=False):
    L, risk, source = load_sector_inputs()
    sol = _solve_with_cvxpy(L, risk, cap5=cap5)
    method = "CVXPY"
    if sol is None or not sol.get("success", False):
        sol = _solve_with_scipy(L, risk, cap5=cap5)
        method = "SciPy/HiGHS"
    if sol is None or not sol.get("success", False):
        # fallback analytic: all xH vào ngành có b1 cao nhất
        i = int(np.argmax(B1))
        xAI = np.zeros(N); xH = np.zeros(N); xH[i] = BUDGET
        sol = {"success": True, "status": "analytic fallback", "xAI": xAI, "xH": xH,
               "objective": B1[i] * BUDGET}
        method = "analytic fallback"
    sol["method"] = method
    sol["data_source"] = source
    sol["L"] = L
    sol["risk"] = risk
    return sol


@st.cache_data(show_spinner=False)
def solve_balanced_by_labor():
    """Kịch bản chính sách bổ sung: chia ngân sách theo tỷ lệ lao động, tối ưu trong từng ngành."""
    L, risk, source = load_sector_inputs()
    budget_i = BUDGET * L / L.sum()
    xAI = np.zeros(N); xH = np.zeros(N)
    net_ai = A1 - C1_DISP * risk
    for i in range(N):
        # Với ngân sách B_i, chọn giữa xH thuần hoặc gói AI+kèm H đủ retrain.
        ratio = (C1_DISP[i] * risk[i]) / D1_CAP[i]
        package_return = (net_ai[i] + B1[i] * ratio) / (1 + ratio)
        if package_return > B1[i]:
            xAI[i] = budget_i[i] / (1 + ratio)
            xH[i] = budget_i[i] - xAI[i]
        else:
            xH[i] = budget_i[i]
    return budget_i, xAI, xH, source


def _metrics(xAI, xH, risk):
    new = A1 * xAI
    displaced = C1_DISP * risk * xAI
    upgrade = B1 * xH
    retrain_cap = D1_CAP * xH
    net = new + upgrade - displaced
    return new, upgrade, displaced, retrain_cap, net


def render():
    bai_header(
        so="9",
        ten="Tác động AI tới thị trường lao động Việt Nam",
        mo_ta="Mô hình NetJob ròng, ngưỡng đào tạo lại, Sankey nhóm dễ tổn thương, ràng buộc mất việc ≤5%",
        cap_do="KHÁ KHÓ",
        tools=["cvxpy", "scipy.optimize", "plotly"],
        thoi_luong="2 tuần",
    )

    L, risk, data_source = load_sector_inputs()
    net_ai = A1 - C1_DISP * risk
    disp_per_ai = C1_DISP * risk
    ratio_retrain = disp_per_ai / D1_CAP

    info_box(
        "Mô hình: <b>NetJobᵢ = NewJobᴬᴵᵢ + UpgradeJobᵢ − DisplacedJobᵢ</b><br>"
        "NewJob = a₁·x_AI · Upgrade = b₁·x_H · Displaced = c₁·riskᵢ·x_AI · RetrainCap = d₁·x_H<br>"
        "Ràng buộc chính: Σ(x_AI+x_H) ≤ 30.000 · NetJobᵢ ≥ 0 · Displacedᵢ ≤ RetrainingCapacityᵢ",
        bg="#E8F5E9", border=C1, icon="📐"
    )
    info_box(
        f"Dữ liệu lao động và risk được đọc từ <b>{data_source}</b>. Hai ngành Khai khoáng và Y tế được loại khỏi bộ 10 ngành theo đúng đề.",
        bg="#E3F2FD", border=CBLUE, icon="📁"
    )

    # ── Bảng tham số ────────────────────────────────────────────────────────
    section_title("Tham số 8 ngành & kiểm tra lợi suất biên", "📊")
    df_param = pd.DataFrame({
        "Ngành": SECTORS_VI,
        "LĐ (triệu)": L,
        "Risk (%)": (risk * 100).round(0).astype(int),
        "a₁ New/tỷ": A1,
        "b₁ Upgrade/tỷ": B1,
        "c₁ Disp/tỷ": C1_DISP,
        "d₁ Cap/tỷ": D1_CAP,
        "Displaced/tỷ AI": disp_per_ai.round(3),
        "Net_AI/tỷ": net_ai.round(3),
        "x_H/x_AI tối thiểu": ratio_retrain.round(3),
    })
    st.dataframe(df_param, use_container_width=True, hide_index=True)

    info_box(
        "Tất cả ngành đều có Net_AI/tỷ dương, nhưng đầu tư AI vẫn phải đi kèm đào tạo nếu phát sinh lao động dịch chuyển. "
        "Hệ số đào tạo cao nhất là Giáo dục (b₁=55), nên mô hình gốc tuyến tính sẽ có nghiệm góc.",
        bg="#FFF8E1", border=CORANGE, icon="💡"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Câu 9.4.1
    # ════════════════════════════════════════════════════════
    section_title("Câu 9.4.1 — Nghiệm tối ưu LP & NetJob ròng", "🎯")
    sol = solve_global(cap5=False)
    xAI, xH = sol["xAI"], sol["xH"]
    new, up, disp, cap, net = _metrics(xAI, xH, risk)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    kpis = [
        (k1, "Tổng NetJob", f"{net.sum():,.0f}", "việc làm ròng", C1),
        (k2, "Ngân sách x_H", f"{xH.sum():,.0f}", "tỷ VND", CPURPLE),
        (k3, "Ngân sách x_AI", f"{xAI.sum():,.0f}", "tỷ VND", CBLUE),
        (k4, "Solver", sol["method"], sol["status"], CORANGE),
    ]
    for col, lbl, val, sub, color in kpis:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:0.78rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
                <div style="font-size:1.55rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.75rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    df_sol = pd.DataFrame({
        "Ngành": SECTORS_VI,
        "x_AI": xAI.round(3),
        "x_H": xH.round(3),
        "NewJob": new.round(0),
        "UpgradeJob": up.round(0),
        "DisplacedJob": disp.round(0),
        "RetrainCap": cap.round(0),
        "NetJob": net.round(0),
    })
    st.dataframe(df_sol, use_container_width=True, hide_index=True)

    top_h = int(np.argmax(xH))
    info_box(
        f"<b>Kết quả toán học của mô hình gốc là nghiệm góc:</b> toàn bộ ngân sách chuyển vào x_H của ngành "
        f"<b>{SECTORS_VI[top_h]}</b>. Lý do: b₁={B1[top_h]:.0f} là lợi suất việc làm lớn nhất và x_H không tạo DisplacedJob. "
        "Kết quả này đúng với mô hình tuyến tính đã cho, nhưng chưa phải khuyến nghị chính sách đầy đủ vì thiếu ràng buộc phủ ngành/tối thiểu triển khai AI.",
        bg="#E3F2FD", border=CBLUE, icon="📌"
    )

    st.markdown("**Kịch bản chính sách bổ sung — chia ngân sách theo tỷ lệ lao động từng ngành rồi tối ưu trong từng ngành:**")
    budget_i, xAI_b, xH_b, _ = solve_balanced_by_labor()
    new_b, up_b, disp_b, cap_b, net_b = _metrics(xAI_b, xH_b, risk)
    df_bal = pd.DataFrame({
        "Ngành": SECTORS_VI,
        "Ngân sách theo LĐ": budget_i.round(0),
        "x_AI": xAI_b.round(0),
        "x_H": xH_b.round(0),
        "Displaced": disp_b.round(0),
        "RetrainCap": cap_b.round(0),
        "NetJob": net_b.round(0),
    })
    st.dataframe(df_bal, use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(
        x=net_b[np.argsort(net_b)],
        y=[SHORT[i] for i in np.argsort(net_b)],
        orientation="h",
        marker_color=[C3 if v >= 0 else CRED for v in net_b[np.argsort(net_b)]],
        text=[f"{v:,.0f}" for v in net_b[np.argsort(net_b)]],
        textposition="outside",
    ))
    fig.update_layout(height=300, margin=dict(l=0, r=60, t=10, b=0), font=CF,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=True, gridcolor="#F0F4F0"), yaxis=dict(autorange="reversed"),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Câu 9.4.2
    # ════════════════════════════════════════════════════════
    section_title("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu ngành CN chế biến chế tạo", "📐")
    i = 1
    net_ai_i = net_ai[i]
    disp_i = disp_per_ai[i]
    net_threshold = 0.0 if net_ai_i >= 0 else (disp_i - A1[i]) / B1[i]
    retrain_ratio = disp_i / D1_CAP[i]
    xAI_max_with_retrain = BUDGET / (1 + retrain_ratio)
    xH_min_for_max_ai = BUDGET - xAI_max_with_retrain

    k1, k2, k3, k4 = st.columns(4, gap="small")
    data2 = [
        (k1, "Net_AI/tỷ", f"{net_ai_i:.3f}", "New − Displaced", C1),
        (k2, "Ngưỡng NetJob≥0", f"{net_threshold:.3f}", "x_H/x_AI", CBLUE),
        (k3, "Ngưỡng retrain", f"{retrain_ratio:.3f}", "x_H/x_AI", CPURPLE),
        (k4, "x_H nếu AI tối đa", f"{xH_min_for_max_ai:,.0f}", "tỷ VND", CORANGE),
    ]
    for col, lbl, val, sub, color in data2:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:0.9rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);text-align:center;">
                <div style="font-size:0.70rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:5px;">{lbl}</div>
                <div style="font-size:1.45rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                <div style="font-size:0.70rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    info_box(
        f"Phải tách hai điều kiện. Nếu chỉ xét <b>NetJob₂ ≥ 0</b>, ngành chế biến có Net_AI = {net_ai_i:.3f} việc/tỷ nên "
        f"x_H tối thiểu bằng 0. Nhưng để bảo đảm <b>Displaced ≤ RetrainingCapacity</b>, cần "
        f"x_H ≥ {retrain_ratio:.3f}·x_AI. Nếu dùng toàn bộ 30.000 tỷ trong ngành 2 với AI ở mức tối đa nhưng vẫn đủ đào tạo, "
        f"x_AI tối đa ≈ {xAI_max_with_retrain:,.0f} tỷ và x_H tối thiểu ≈ {xH_min_for_max_ai:,.0f} tỷ.",
        bg="#FFF8E1", border=CORANGE, icon="💡"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Câu 9.4.3
    # ════════════════════════════════════════════════════════
    section_title("Câu 9.4.3 — Sankey nhóm dễ tổn thương", "🌊")
    vuln = [0, 2, 3]  # NLTS, Xây dựng, Bán buôn-bán lẻ
    xAI_demo = 2000.0
    labels = ["Đầu tư AI"] + [SHORT[i] for i in vuln] + ["Việc làm mới", "Lao động dịch chuyển", "Đào tạo lại", "Việc nâng cấp"]
    idx_invest = 0
    idx_new = 1 + len(vuln)
    idx_disp = idx_new + 1
    idx_retrain = idx_new + 2
    idx_up = idx_new + 3
    src, tgt, val, colors = [], [], [], []
    for k, i in enumerate(vuln):
        node = 1 + k
        new_i = A1[i] * xAI_demo
        disp_i_demo = disp_per_ai[i] * xAI_demo
        xH_need = disp_i_demo / D1_CAP[i]
        upgrade_i = B1[i] * xH_need
        src += [idx_invest, node, node, idx_disp, idx_retrain]
        tgt += [node, idx_new, idx_disp, idx_retrain, idx_up]
        val += [xAI_demo, new_i, disp_i_demo, disp_i_demo, upgrade_i]
        colors += ["rgba(25,118,210,0.35)", "rgba(26,107,60,0.45)", "rgba(229,57,53,0.35)",
                   "rgba(123,31,162,0.35)", "rgba(46,139,87,0.35)"]

    fig_sk = go.Figure(go.Sankey(
        node=dict(pad=18, thickness=18, label=labels,
                  color=[CBLUE] + [CORANGE] * len(vuln) + [C1, CRED, CPURPLE, C2],
                  line=dict(color="white", width=1)),
        link=dict(source=src, target=tgt, value=val, color=colors)
    ))
    fig_sk.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0), font=CF,
                         paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sk, use_container_width=True, config={"displayModeBar": False})

    info_box(
        "Sankey chỉ là mô phỏng minh họa cho 3 ngành dễ tổn thương. Bán buôn-bán-lẻ tạo dòng dịch chuyển lớn nhất trong nhóm này do risk và c₁ cao. "
        "Điểm chính sách là phải gắn triển khai AI với năng lực đào tạo lại tương ứng, không chỉ nhìn số việc làm mới.",
        bg="#E8F5E9", border=C3, icon="✅"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Câu 9.4.4
    # ════════════════════════════════════════════════════════
    section_title("Câu 9.4.4 — Thêm ràng buộc không ngành nào mất quá 5% lao động", "🛡️")
    sol5 = solve_global(cap5=True)
    xAI5, xH5 = sol5["xAI"], sol5["xH"]
    new5, up5, disp5, cap5_arr, net5 = _metrics(xAI5, xH5, risk)

    c1, c2, c3 = st.columns(3, gap="medium")
    vals = [
        (c1, "Khả thi", "✅ Có", sol5["method"], C1),
        (c2, "Tổng NetJob", f"{net5.sum():,.0f}", "việc làm ròng", C1),
        (c3, "Displaced max", f"{disp5.max():,.0f}", "việc", CRED if disp5.max() > 0 else C2),
    ]
    for col, lbl, valtxt, sub, color in vals:
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                        padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:0.78rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
                <div style="font-size:1.55rem;font-weight:900;color:{color};line-height:1;">{valtxt}</div>
                <div style="font-size:0.75rem;color:#8AA898;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    df5 = pd.DataFrame({
        "Ngành": SECTORS_VI,
        "Displaced": disp5.round(0),
        "Giới hạn 5% LĐ": (0.05 * L * 1_000_000).round(0),
        "x_AI": xAI5.round(2),
        "x_H": xH5.round(2),
        "NetJob": net5.round(0),
    })
    st.dataframe(df5, use_container_width=True, hide_index=True)

    info_box(
        "Bài toán vẫn khả thi. Với mô hình gốc, nghiệm tối ưu đã chọn đào tạo lại thuần túy nên Displaced≈0; "
        "vì vậy ràng buộc 5% không làm giảm mục tiêu. Ràng buộc này sẽ trở nên quan trọng nếu nhà nước áp thêm mức tối thiểu đầu tư AI theo ngành.",
        bg="#E8F5E9", border=C1, icon="🛡️"
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ── Thảo luận ───────────────────────────────────────────────────────────
    section_title("Câu hỏi thảo luận chính sách", "💬")
    q_data = [
        ("a)", "Ngành nào cần đầu tư đào tạo lại nhiều nhất? Có khớp thực tế Việt Nam không?",
         "Theo nghiệm LP gốc, Giáo dục nhận toàn bộ ngân sách vì b₁=55 cao nhất. Đây là kết quả toán học, không phải phương án phân bổ xã hội tối ưu. "
         "Nếu thêm tiêu chí phủ ngành theo quy mô lao động, Nông-Lâm-Thủy sản và CN chế biến cần ngân sách đào tạo lớn vì có số lao động rất đông."),
        ("b)", "Tài chính-Ngân hàng có risk 52% nhưng hệ số tạo việc mới cao. Mô hình khuyến nghị gì?",
         f"Tài chính có Net_AI = {net_ai[4]:.2f} việc/tỷ, vẫn dương nhưng bị kéo xuống bởi risk cao. Khuyến nghị là triển khai AI có kiểm soát, kèm đào tạo lại bắt buộc: "
         f"x_H ≥ {ratio_retrain[4]:.3f}·x_AI để bảo đảm năng lực hấp thụ."),
        ("c)", "Có nên đầu tư x_AI vào Nông-Lâm-Thủy sản không?",
         f"Có thể đầu tư nhưng không nên ưu tiên AI thuần túy. Net_AI của nông nghiệp = {net_ai[0]:.2f} việc/tỷ, thấp hơn nhiều so với đào tạo H = 45 việc/tỷ. "
         "Mô hình nghiêng về đào tạo kỹ năng, số hóa quy trình và hỗ trợ nông dân trước khi tự động hóa mạnh."),
        ("d)", "'Tốc độ tự động hóa không vượt quá năng lực đào tạo lại' là ràng buộc nào?",
         "Chính là DisplacedJobᵢ ≤ RetrainingCapacityᵢ, tức c₁ᵢ·riskᵢ·x_AIᵢ ≤ d₁ᵢ·x_Hᵢ. "
         "Nên bổ sung thêm trần mất việc theo tỷ lệ lao động, quỹ hỗ trợ chuyển đổi nghề, và ràng buộc đào tạo hoàn thành trước khi triển khai AI quy mô lớn."),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()
