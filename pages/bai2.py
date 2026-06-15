"""pages/bai2.py — Bài 2: Phân bổ ngân sách số theo 4 hạng mục đầu tư

Bản làm lại:
- Giải bài toán LP bằng scipy.optimize.linprog.
- Giải lại bằng PuLP/CBC và lấy giá đối ngẫu `.pi` cho từng ràng buộc.
- Nếu môi trường chưa cài PuLP, dùng shadow price từ SciPy/HiGHS để vẫn hiển thị đúng kết quả.
- Phân tích độ nhạy B = 100, 120, 140 và kịch bản x3 >= 30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

try:
    import pulp
    HAS_PULP = True
except Exception:  # pragma: no cover
    HAS_PULP = False

from utils import bai_header, end_padding, info_box, section_title


# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"
C2 = "#2E8B57"
C3 = "#4CAF72"
CBLUE = "#1976D2"
CRED = "#E53935"
CORANGE = "#E65100"
CPURPLE = "#7B1FA2"
CGREY = "#6B8A7A"
CF = dict(family="Montserrat, sans-serif", size=12)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DATA
# ─────────────────────────────────────────────────────────────────────────────
VAR_CODES = ["x₁", "x₂", "x₃", "x₄"]
VAR_NAMES = ["Hạ tầng số", "AI & dữ liệu", "Nhân lực số", "R&D công nghệ"]
VAR_LABELS = [f"{c} — {n}" for c, n in zip(VAR_CODES, VAR_NAMES)]
OBJ = np.array([0.85, 1.20, 0.95, 1.35], dtype=float)
LB_BASE = np.array([25, 15, 20, 10], dtype=float)

# Dạng scipy: minimize -Z, mọi ràng buộc về dạng A_ub x <= b_ub
# Công nghệ chiến lược:
# x2 + x4 >= 0.35(x1+x2+x3+x4)
# <=> 0.35x1 - 0.65x2 + 0.35x3 - 0.65x4 <= 0
A_UB_TEMPLATE = np.array([
    [1.00,  1.00, 1.00,  1.00],   # ngân sách tổng <= B
    [-1.0,  0.00, 0.00,  0.00],   # x1 >= lb1
    [0.00, -1.00, 0.00,  0.00],   # x2 >= lb2
    [0.00,  0.00,-1.00,  0.00],   # x3 >= lb3
    [0.00,  0.00, 0.00, -1.00],   # x4 >= lb4
    [0.35, -0.65, 0.35, -0.65],   # tỷ trọng AI+R&D >= 35%
], dtype=float)

CONSTRAINT_LABELS = [
    "Ngân sách tổng",
    "Sàn x₁ — Hạ tầng số",
    "Sàn x₂ — AI & dữ liệu",
    "Sàn x₃ — Nhân lực số",
    "Sàn x₄ — R&D công nghệ",
    "Tỷ trọng công nghệ chiến lược",
]

CONSTRAINT_FORMS = [
    "x₁+x₂+x₃+x₄ ≤ B",
    "x₁ ≥ 25",
    "x₂ ≥ 15",
    "x₃ ≥ 20",
    "x₄ ≥ 10",
    "x₂+x₄ ≥ 35% tổng ngân sách",
]

CONSTRAINT_EXPLAIN = [
    "+1 nghìn tỷ ngân sách làm Z* tăng theo hệ số biên tốt nhất còn khả dụng.",
    "+1 nghìn tỷ sàn hạ tầng làm giảm Z* vì vốn phải rời khỏi R&D.",
    "+1 nghìn tỷ sàn AI làm giảm Z* nhẹ vì AI kém R&D 0,15 điểm.",
    "+1 nghìn tỷ sàn nhân lực làm giảm Z* vì nhân lực kém R&D 0,40 điểm.",
    "Không ảnh hưởng tại nghiệm vì R&D đang nhận 40, cao hơn sàn 10.",
    "Không ảnh hưởng tại nghiệm vì AI+R&D = 55%, cao hơn mức yêu cầu 35%.",
]


# ─────────────────────────────────────────────────────────────────────────────
# SOLVERS
# ─────────────────────────────────────────────────────────────────────────────
def _analytic_solution(B: float = 100.0, x3_min: float = 20.0) -> dict:
    """Fallback giải tay cho cấu trúc bài này.

    Vì R&D có hệ số lớn nhất, nghiệm tối ưu đặt các biến còn lại ở sàn,
    rồi dồn toàn bộ phần dư vào x4 nếu ràng buộc 35% không binding.
    """
    lb = LB_BASE.copy()
    lb[2] = x3_min
    remaining = B - lb.sum()
    if remaining < -1e-9:
        return {"success": False, "status": "Infeasible", "message": "Tổng sàn vượt ngân sách", "x": None, "Z": None}

    x = lb.copy()
    x[3] += remaining
    tech_left = x[1] + x[3]
    tech_req = 0.35 * x.sum()
    if tech_left + 1e-9 < tech_req:
        return {"success": False, "status": "Infeasible", "message": "Không thỏa tỷ trọng công nghệ chiến lược", "x": None, "Z": None}

    Z = float(OBJ @ x)
    return {"success": True, "status": "Optimal", "message": "Analytic fallback", "x": x, "Z": Z}


@st.cache_data(show_spinner=False)
def solve_scipy(B: float = 100.0, x3_min: float = 20.0) -> dict:
    """Giải bằng scipy.optimize.linprog/HiGHS."""
    if not HAS_SCIPY:
        return _analytic_solution(B, x3_min)

    lb = LB_BASE.copy()
    lb[2] = x3_min
    b_ub = np.array([B, -lb[0], -lb[1], -lb[2], -lb[3], 0.0], dtype=float)
    res = linprog(
        c=-OBJ,
        A_ub=A_UB_TEMPLATE,
        b_ub=b_ub,
        bounds=[(0, None)] * 4,
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "status": str(res.status),
            "message": res.message,
            "x": None,
            "Z": None,
            "slack": None,
            "marginals_min": None,
            "shadow_economic": None,
        }

    # SciPy đang giải bài toán minimize(-Z).
    # - Với ngân sách dạng <= B: shadow của max Z = - marginal_min.
    # - Với các ràng buộc sàn đã đổi thành -x_i <= -lb_i:
    #   shadow theo việc tăng mức sàn lb_i = marginal_min.
    # - Với ràng buộc công nghệ dạng <=: shadow của max Z = - marginal_min,
    #   nhưng tại nghiệm bài này non-binding nên bằng 0.
    marg = np.array(res.ineqlin.marginals, dtype=float)
    shadow_economic = np.array([
        -marg[0],   # ngân sách
        marg[1],    # tăng lb x1
        marg[2],    # tăng lb x2
        marg[3],    # tăng lb x3
        marg[4],    # tăng lb x4
        -marg[5],   # làm chặt ràng buộc công nghệ trong dạng <=
    ], dtype=float)

    return {
        "success": True,
        "status": "Optimal",
        "message": res.message,
        "x": np.array(res.x, dtype=float),
        "Z": float(-res.fun),
        "slack": np.array(res.ineqlin.residual, dtype=float),
        "marginals_min": marg,
        "shadow_economic": shadow_economic,
    }


@st.cache_data(show_spinner=False)
def solve_pulp(B: float = 100.0, x3_min: float = 20.0) -> dict:
    """Giải lại bằng PuLP/CBC và lấy dual values `.pi` của từng ràng buộc."""
    if not HAS_PULP:
        return {"installed": False, "success": False, "message": "PuLP chưa được cài trong môi trường hiện tại."}

    m = pulp.LpProblem("VN_Digital_Budget_4_Items", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1_infrastructure", lowBound=0)
    x2 = pulp.LpVariable("x2_ai_data", lowBound=0)
    x3 = pulp.LpVariable("x3_digital_human_capital", lowBound=0)
    x4 = pulp.LpVariable("x4_rd_technology", lowBound=0)
    xs = [x1, x2, x3, x4]

    m += 0.85 * x1 + 1.20 * x2 + 0.95 * x3 + 1.35 * x4, "GDP_gain"

    # Đặt tên ràng buộc để lấy m.constraints[name].pi sau khi CBC giải xong.
    m += x1 + x2 + x3 + x4 <= B, "C1_budget_total"
    m += x1 >= 25, "C2_min_infrastructure"
    m += x2 >= 15, "C3_min_ai_data"
    m += x3 >= x3_min, "C4_min_human_capital"
    m += x4 >= 10, "C5_min_rd"
    m += x2 + x4 >= 0.35 * (x1 + x2 + x3 + x4), "C6_strategic_technology_share"

    try:
        solver = pulp.PULP_CBC_CMD(msg=0)
        m.solve(solver)
    except Exception as exc:  # pragma: no cover
        return {"installed": True, "success": False, "message": f"Không chạy được CBC: {exc}"}

    status = pulp.LpStatus.get(m.status, str(m.status))
    success = status == "Optimal"
    if not success:
        return {"installed": True, "success": False, "status": status, "message": "PuLP/CBC không tìm được nghiệm tối ưu."}

    x_val = np.array([pulp.value(v) for v in xs], dtype=float)
    Z = float(pulp.value(m.objective))

    rows = []
    for cname, label, form in zip(m.constraints.keys(), CONSTRAINT_LABELS, CONSTRAINT_FORMS):
        con = m.constraints[cname]
        rows.append({
            "Tên ràng buộc PuLP": cname,
            "Ràng buộc": label,
            "Dạng": form,
            "Slack PuLP": float(con.slack),
            "Dual/Pi PuLP": float(con.pi),
        })

    return {
        "installed": True,
        "success": True,
        "status": status,
        "message": "Solved by PuLP/CBC",
        "x": x_val,
        "Z": Z,
        "dual_raw": pd.DataFrame(rows),
    }


@st.cache_data(show_spinner=False)
def sensitivity_table() -> pd.DataFrame:
    """Tạo bảng độ nhạy cho nhiều mức ngân sách."""
    records = []
    for B in range(100, 165, 5):
        res = solve_scipy(float(B), 20.0)
        if res["success"]:
            records.append({
                "Ngân sách B": B,
                "x₁": res["x"][0],
                "x₂": res["x"][1],
                "x₃": res["x"][2],
                "x₄": res["x"][3],
                "Z*": res["Z"],
            })
    return pd.DataFrame(records)


def dual_table_from_scipy(res: dict) -> pd.DataFrame:
    """Chuẩn hóa bảng shadow price từ SciPy để dùng khi PuLP không có."""
    slack = res.get("slack")
    shadow = res.get("shadow_economic")
    if slack is None or shadow is None:
        return pd.DataFrame()

    rows = []
    for i, (label, form, expl) in enumerate(zip(CONSTRAINT_LABELS, CONSTRAINT_FORMS, CONSTRAINT_EXPLAIN)):
        rows.append({
            "Ràng buộc": label,
            "Dạng": form,
            "Slack": float(slack[i]),
            "Shadow price kinh tế": float(shadow[i]),
            "Trạng thái": "Binding" if abs(slack[i]) < 1e-7 else "Non-binding",
            "Diễn giải": expl,
        })
    return pd.DataFrame(rows)


def compare_pulp_scipy(base: dict, pulp_res: dict) -> tuple[str, str]:
    """Tạo thông báo so sánh PuLP và SciPy."""
    if not pulp_res.get("installed", False):
        return "PuLP chưa cài", "Ứng dụng đang hiển thị nghiệm SciPy/HiGHS và shadow price chuẩn hóa từ SciPy. Khi cài PuLP, bảng dual `.pi` sẽ tự động hiện."
    if not pulp_res.get("success", False):
        return "PuLP chưa có nghiệm", pulp_res.get("message", "Không có thông tin lỗi.")

    diff_x = float(np.max(np.abs(base["x"] - pulp_res["x"])))
    diff_z = abs(float(base["Z"]) - float(pulp_res["Z"]))
    if diff_x < 1e-6 and diff_z < 1e-6:
        return "PuLP/CBC ≡ SciPy/HiGHS", "Hai solver cho cùng nghiệm tối ưu; sai khác chỉ ở mức làm tròn số học."
    return "Có sai khác nhỏ", f"max|Δx| = {diff_x:.3e}; |ΔZ| = {diff_z:.3e}."


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render():
    bai_header(
        so="2",
        ten="Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số",
        mo_ta="Giải LP bằng scipy.optimize.linprog và PuLP/CBC; đọc shadow price, phân tích độ nhạy ngân sách",
        cap_do="DỄ",
        tools=["scipy.optimize", "pulp", "plotly"],
        thoi_luong="1 tuần",
    )

    base = solve_scipy(100.0, 20.0)
    pulp_base = solve_pulp(100.0, 20.0)
    sens = sensitivity_table()
    x330 = solve_scipy(100.0, 30.0)

    if not base["success"]:
        info_box(
            f"Không giải được bài toán gốc: {base.get('message', '')}",
            bg="#FCE4EC", border=CRED, icon="⚠️",
        )
        return

    sol = base["x"]
    Z = base["Z"]

    # ── Model summary ────────────────────────────────────────────────────────
    info_box(
        "Hàm mục tiêu: <b>max Z = 0,85·x₁ + 1,20·x₂ + 0,95·x₃ + 1,35·x₄</b> "
        "với ngân sách tổng 100 nghìn tỷ VND và tỷ trọng AI+R&D tối thiểu 35%.",
        bg="#E8F5E9", border=C1, icon="📐",
    )

    col_var, col_con = st.columns([1, 1], gap="large")
    with col_var:
        st.markdown(
            """
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                        padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);height:100%;">
                <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:0.8rem;">
                    🔧 Biến quyết định & hệ số tác động</div>
            """,
            unsafe_allow_html=True,
        )
        df_var = pd.DataFrame({
            "Biến": VAR_CODES,
            "Hạng mục": VAR_NAMES,
            "Hệ số GDP": OBJ,
            "Sàn tối thiểu": LB_BASE,
        })
        st.dataframe(df_var, use_container_width=True, hide_index=True)
        st.markdown(
            "<div style='font-size:0.82rem;color:#6B8A7A;margin-top:0.5rem;'>"
            "R&D có hệ số cao nhất 1,35 nên phần ngân sách dư sẽ chảy vào R&D nếu các ràng buộc khác không binding."
            "</div></div>",
            unsafe_allow_html=True,
        )

    with col_con:
        st.markdown(
            """
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                        padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);height:100%;">
                <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:0.8rem;">
                    📏 Hệ ràng buộc</div>
            """,
            unsafe_allow_html=True,
        )
        for con in [
            "x₁ + x₂ + x₃ + x₄ ≤ 100",
            "x₁ ≥ 25; x₂ ≥ 15; x₃ ≥ 20; x₄ ≥ 10",
            "x₂ + x₄ ≥ 0,35·(x₁+x₂+x₃+x₄)",
            "x₁, x₂, x₃, x₄ ≥ 0",
        ]:
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.82rem;"
                f"padding:7px 12px;background:#F4F6F8;border-radius:8px;margin:5px 0;"
                f"color:#1A2B1F;'>{con}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 2.4.1 & 2.4.2
    # ════════════════════════════════════════════════════════
    section_title("Câu 2.4.1 & 2.4.2 — Nghiệm tối ưu, PuLP và shadow price", "🎯")

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    kpis = [
        (k1, "Z* tối ưu", f"{Z:.2f}", "nghìn tỷ GDP", C1),
        (k2, "x₁ Hạ tầng", f"{sol[0]:.0f}", "đúng mức sàn", CGREY),
        (k3, "x₂ AI", f"{sol[1]:.0f}", "đúng mức sàn", CGREY),
        (k4, "x₃ Nhân lực", f"{sol[2]:.0f}", "đúng mức sàn", CGREY),
        (k5, "x₄ R&D", f"{sol[3]:.0f}", "nhận phần dư", CORANGE),
    ]
    for col, lbl, val, sub, color in kpis:
        with col:
            st.markdown(
                f"""
                <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                            padding:0.9rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);text-align:center;">
                    <div style="font-size:0.73rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:5px;">{lbl}</div>
                    <div style="font-size:1.6rem;font-weight:900;color:{color};line-height:1;">{val}</div>
                    <div style="font-size:0.73rem;color:#8AA898;margin-top:4px;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    col_chart, col_dual = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown(
            """
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                        padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
                <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:3px;">
                    Phân bổ ngân sách tối ưu bằng SciPy/linprog</div>
                <div style="font-size:0.8rem;color:#8AA898;margin-bottom:0.8rem;">
                    x₁, x₂, x₃ ở mức sàn; phần dư 40 nghìn tỷ dồn vào R&D vì hệ số 1,35 cao nhất.</div>
            """,
            unsafe_allow_html=True,
        )
        fig = go.Figure(go.Bar(
            x=VAR_LABELS,
            y=sol,
            marker_color=[C3, CBLUE, C2, CORANGE],
            marker_line_width=0,
            text=[f"{v:.0f}" for v in sol],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.0f} nghìn tỷ VND<extra></extra>",
        ))
        fig.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=15, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=CF,
            showlegend=False,
            bargap=0.35,
            yaxis=dict(showgrid=True, gridcolor="#F0F4F0", zeroline=False, range=[0, 48], title="Nghìn tỷ VND"),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_dual:
        title, msg = compare_pulp_scipy(base, pulp_base)
        st.markdown(
            f"""
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                        padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);height:100%;">
                <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:0.8rem;">
                    ✅ So sánh solver</div>
                <div style="background:#E8F5E9;border-radius:10px;padding:0.9rem 1rem;margin-bottom:0.8rem;">
                    <div style="font-size:0.82rem;color:#6B8A7A;font-weight:700;text-transform:uppercase;">{title}</div>
                    <div style="font-size:0.82rem;color:#1A2B1F;margin-top:4px;">{msg}</div>
                </div>
                <div style="background:linear-gradient(135deg,#1A6B3C,#2E8B57);border-radius:12px;
                            padding:1.1rem 1.2rem;color:#fff;">
                    <div style="font-size:0.78rem;opacity:0.82;text-transform:uppercase;letter-spacing:0.05em;">
                        Shadow price ngân sách tổng</div>
                    <div style="font-size:2.2rem;font-weight:900;line-height:1.1;margin:4px 0;">1,35</div>
                    <div style="font-size:0.82rem;opacity:0.88;">
                        +1 nghìn tỷ ngân sách → +1,35 nghìn tỷ GDP gain</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # Bảng kết quả và dual đầy đủ
    alloc_df = pd.DataFrame({
        "Biến": VAR_CODES,
        "Hạng mục": VAR_NAMES,
        "Giá trị tối ưu": np.round(sol, 4),
        "Hệ số mục tiêu": OBJ,
        "Đóng góp vào Z*": np.round(sol * OBJ, 4),
    })
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("**Bảng giá đối ngẫu từng ràng buộc**")

    dual_df = dual_table_from_scipy(base)
    if not dual_df.empty:
        st.dataframe(
            dual_df[["Ràng buộc", "Dạng", "Slack", "Shadow price kinh tế", "Trạng thái", "Diễn giải"]],
            use_container_width=True,
            hide_index=True,
        )

    if pulp_base.get("installed") and pulp_base.get("success"):
        with st.expander("Xem dual raw từ PuLP/CBC (.pi, .slack)"):
            st.dataframe(pulp_base["dual_raw"], use_container_width=True, hide_index=True)
    else:
        info_box(
            "Máy hiện tại chưa cài PuLP hoặc CBC không khả dụng, nên bảng dual đang lấy từ SciPy/HiGHS. "
            "Khi cài PuLP, app sẽ tự hiển thị thêm bảng `.pi` raw của PuLP.",
            bg="#FFF8E1", border=CORANGE, icon="ℹ️",
        )

    info_box(
        "Diễn giải chính sách: shadow price ngân sách tổng = <b>1,35</b>. Nghĩa là tại nghiệm tối ưu, "
        "nếu ngân sách tăng thêm 1 nghìn tỷ VND, phần tăng thêm sẽ được phân bổ vào R&D và làm GDP kỳ vọng "
        "tăng thêm 1,35 nghìn tỷ VND. Các ràng buộc sàn x₁, x₂, x₃ có shadow price âm vì nếu ép tăng sàn, "
        "một phần vốn phải rời khỏi R&D nên Z* giảm.",
        bg="#E3F2FD", border=CBLUE, icon="📋",
    )

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 2.4.3
    # ════════════════════════════════════════════════════════
    section_title("Câu 2.4.3 — Phân tích độ nhạy khi tăng ngân sách", "📈")

    col_sens, col_tbl = st.columns([3, 2], gap="large")
    with col_sens:
        st.markdown(
            """
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                        padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
                <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:3px;">
                    Đường cong Z*(B)</div>
                <div style="font-size:0.8rem;color:#8AA898;margin-bottom:0.8rem;">
                    Trong khoảng khảo sát, nghiệm giữ cấu trúc cũ nên đường cong tuyến tính với độ dốc 1,35.</div>
            """,
            unsafe_allow_html=True,
        )
        fig2 = go.Figure(go.Scatter(
            x=sens["Ngân sách B"],
            y=sens["Z*"],
            mode="lines+markers",
            line=dict(color=C1, width=3),
            marker=dict(size=7, color=C1, line=dict(width=2, color="#fff")),
            fill="tozeroy",
            fillcolor="rgba(26,107,60,0.06)",
            hovertemplate="B=%{x}<br>Z*=%{y:.2f}<extra></extra>",
        ))
        for B in [100, 120, 140]:
            z_val = float(sens.loc[sens["Ngân sách B"] == B, "Z*"].iloc[0])
            fig2.add_annotation(
                x=B,
                y=z_val,
                text=f"{z_val:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-25,
                font=dict(size=10, color=CORANGE),
                arrowcolor=CORANGE,
            )
        fig2.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=CF,
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor="#F0F4F0", title="Z*"),
            xaxis=dict(showgrid=False, title="Ngân sách B"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_tbl:
        st.markdown(
            """
            <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                        padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);height:100%;">
                <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:0.8rem;">
                    Bảng tại các mốc đề bài</div>
            """,
            unsafe_allow_html=True,
        )
        focus = sens[sens["Ngân sách B"].isin([100, 120, 140])].copy()
        focus["ΔZ so với mốc trước"] = ["-", "+27.00", "+27.00"]
        st.dataframe(
            focus[["Ngân sách B", "x₁", "x₂", "x₃", "x₄", "Z*", "ΔZ so với mốc trước"]],
            use_container_width=True,
            hide_index=True,
        )
        info_box(
            "B = 100 → Z* = 112,25; B = 120 → Z* = 139,25; B = 140 → Z* = 166,25. "
            "Mỗi +20 ngân sách tạo thêm +27 GDP gain.",
            bg="#E8F5E9", border=C3, icon="💡",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 2.4.4
    # ════════════════════════════════════════════════════════
    section_title("Câu 2.4.4 — Kịch bản ưu tiên nhân lực số: x₃ ≥ 30", "⚖️")

    if not x330["success"]:
        info_box(
            "Kịch bản x₃ ≥ 30 không khả thi vì tổng sàn vượt ngân sách hoặc vi phạm ràng buộc khác.",
            bg="#FCE4EC", border=CRED, icon="⚠️",
        )
    else:
        col_cmp, col_note = st.columns([3, 2], gap="large")
        with col_cmp:
            st.markdown(
                """
                <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                            padding:1.2rem 1.4rem 0.5rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);">
                    <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:3px;">
                        So sánh phân bổ: bài gốc vs x₃ ≥ 30</div>
                    <div style="font-size:0.8rem;color:#8AA898;margin-bottom:0.8rem;">
                        Tăng sàn nhân lực từ 20 lên 30 làm 10 nghìn tỷ dịch chuyển từ R&D sang nhân lực số.</div>
                """,
                unsafe_allow_html=True,
            )
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name="Bài gốc",
                x=VAR_CODES,
                y=base["x"],
                marker_color=C1,
                marker_line_width=0,
                text=[f"{v:.0f}" for v in base["x"]],
                textposition="outside",
            ))
            fig3.add_trace(go.Bar(
                name="x₃ ≥ 30",
                x=VAR_CODES,
                y=x330["x"],
                marker_color=CORANGE,
                marker_line_width=0,
                text=[f"{v:.0f}" for v in x330["x"]],
                textposition="outside",
            ))
            fig3.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=15, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=CF,
                barmode="group",
                bargap=0.32,
                legend=dict(orientation="h", y=-0.18),
                yaxis=dict(showgrid=True, gridcolor="#F0F4F0", zeroline=False, range=[0, 48]),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        loss = float(base["Z"] - x330["Z"])
        with col_note:
            st.markdown(
                f"""
                <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:14px;
                            padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(26,107,60,0.06);height:100%;">
                    <div style="font-weight:700;font-size:0.93rem;color:#1A2B1F;margin-bottom:0.8rem;">
                        Tác động tới Z*</div>
                    <div style="display:flex;justify-content:space-between;padding:0.7rem 0;border-bottom:1px solid #F0F4F0;">
                        <span style="font-size:0.85rem;color:#6B8A7A;">Z* bài gốc</span>
                        <span style="font-size:1.2rem;font-weight:800;color:{C1};">{base['Z']:.2f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:0.7rem 0;border-bottom:1px solid #F0F4F0;">
                        <span style="font-size:0.85rem;color:#6B8A7A;">Z* khi x₃≥30</span>
                        <span style="font-size:1.2rem;font-weight:800;color:{CORANGE};">{x330['Z']:.2f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:0.9rem 0 0;">
                        <span style="font-size:0.85rem;color:#6B8A7A;font-weight:600;">Mức giảm Z*</span>
                        <span style="font-size:1.4rem;font-weight:900;color:{CRED};">−{loss:.2f}</span>
                    </div>
                    <div style="background:#FFF3E0;border-radius:8px;padding:0.75rem 0.9rem;margin-top:0.8rem;">
                        <div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#1A2B1F;">
                            (1,35 − 0,95) × 10 = <b>4,00</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        info_box(
            "Bài toán vẫn <b>khả thi</b>. Nghiệm mới là x₁=25, x₂=15, x₃=30, x₄=30; "
            "Z* giảm từ 112,25 xuống 108,25. Đây là chi phí hiệu quả khi ưu tiên nhân lực số: "
            "10 nghìn tỷ chuyển từ R&D sang H, mất 0,40 điểm hệ số trên mỗi đơn vị.",
            bg="#FFF8E1", border=CORANGE, icon="💡",
        )

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # POLICY DISCUSSION
    # ════════════════════════════════════════════════════════
    section_title("Câu hỏi thảo luận chính sách", "💬")
    q_data = [
        (
            "a)",
            "Khi ngân sách tổng tăng thêm 1 tỷ VND, GDP kỳ vọng tăng thêm bao nhiêu?",
            "Theo shadow price của ràng buộc ngân sách tổng, mỗi 1 nghìn tỷ VND tăng thêm làm GDP kỳ vọng tăng "
            "1,35 nghìn tỷ VND. Đây là cận trên hợp lý của chi phí cơ hội vốn công trong phạm vi mô hình, vì đồng vốn "
            "biên đang được đưa vào hạng mục có hệ số cao nhất là R&D.",
        ),
        (
            "b)",
            "Vì sao R&D có hệ số tác động cao nhất nhưng ràng buộc tối thiểu thấp nhất?",
            "Hệ số cao phản ánh lợi ích kỳ vọng và tác động lan tỏa dài hạn, còn ràng buộc tối thiểu thấp phản ánh "
            "rủi ro triển khai, độ trễ tác động và năng lực hấp thụ thực tế. R&D chỉ hiệu quả khi có hạ tầng số, dữ liệu "
            "và nhân lực đủ tốt, nên chính sách không nên ép mức sàn quá cao ngay từ đầu.",
        ),
        (
            "c)",
            "Tỷ lệ 35% cho AI + R&D có khả thi không?",
            "Trong nghiệm tối ưu, AI+R&D chiếm 55% ngân sách nên ràng buộc 35% không binding về mặt toán học. "
            "Tuy nhiên trong thực tiễn quản lý ngân sách, mức 35% vẫn cần cân nhắc vì ngân sách nhà nước còn chịu áp lực "
            "từ hạ tầng giao thông, an sinh, y tế, giáo dục và chi thường xuyên. Do đó đây là mục tiêu định hướng, cần lộ trình "
            "giải ngân và cơ chế đánh giá hiệu quả rõ ràng.",
        ),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()
