"""pages/bai5.py — Bài 5: MIP lựa chọn dự án chuyển đổi số quốc gia

Phiên bản làm lại:
- Giải đúng MIP bằng PuLP/CBC nếu có.
- Có fallback brute-force exact enumeration vì chỉ có 15 biến nhị phân.
- Xử lý tie-break chặt chẽ: tối đa hóa NPV trước, sau đó chọn nghiệm có chi phí thấp hơn.
- Câu 5.4.2 tách rõ 2 cách hiểu: chỉ nới ngân sách tổng lên 100.000, hoặc đồng thời nới ngân sách năm 1-2 lên 50.000.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import bai_header, end_padding, info_box, section_title

try:
    from pulp import (
        LpProblem,
        LpMaximize,
        LpMinimize,
        LpVariable,
        lpSum,
        value,
        LpStatus,
        PULP_CBC_CMD,
    )
    HAS_PULP = True
except Exception:
    HAS_PULP = False

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C1 = "#1A6B3C"; C2 = "#2E8B57"; C3 = "#4CAF72"
CBLUE = "#1976D2"; CRED = "#E53935"; CORANGE = "#E65100"; CPURPLE = "#7B1FA2"
CF = dict(family="Montserrat, sans-serif", size=12)

# ── Dữ liệu đề bài ────────────────────────────────────────────────────────────
PROJECTS = list(range(1, 16))
DATA = {
    1: {"name":"Trung tâm dữ liệu quốc gia Hòa Lạc", "sector":"Hạ tầng", "C":12000, "C12":8500, "B":21500},
    2: {"name":"Trung tâm dữ liệu quốc gia phía Nam", "sector":"Hạ tầng", "C":11500, "C12":7500, "B":20800},
    3: {"name":"Hệ thống 5G phủ sóng toàn quốc", "sector":"Hạ tầng", "C":18000, "C12":12000, "B":32500},
    4: {"name":"Hệ thống định danh điện tử VNeID 2.0", "sector":"Chính phủ số", "C":4500, "C12":3500, "B":9200},
    5: {"name":"Cổng dịch vụ công quốc gia v3", "sector":"Chính phủ số", "C":3200, "C12":2500, "B":6800},
    6: {"name":"Y tế số quốc gia", "sector":"Y tế số", "C":5800, "C12":4000, "B":11400},
    7: {"name":"Giáo dục số K-12 toàn quốc", "sector":"Giáo dục", "C":6500, "C12":4500, "B":12200},
    8: {"name":"Trung tâm AI quốc gia + supercomputing", "sector":"AI", "C":15000, "C12":9000, "B":28500},
    9: {"name":"Sandbox tài chính số", "sector":"Tài chính số", "C":2500, "C12":1800, "B":5800},
    10:{"name":"Logistics thông minh + cảng biển số", "sector":"Logistics", "C":7200, "C12":5000, "B":13800},
    11:{"name":"Nông nghiệp số ĐBSCL", "sector":"Nông nghiệp", "C":4800, "C12":3500, "B":8500},
    12:{"name":"Đào tạo 50.000 kỹ sư AI/bán dẫn", "sector":"Nhân lực", "C":8500, "C12":5500, "B":16200},
    13:{"name":"Khu CN bán dẫn Bắc Ninh - Bắc Giang", "sector":"Bán dẫn", "C":20000, "C12":13000, "B":35000},
    14:{"name":"An ninh mạng quốc gia (SOC)", "sector":"An ninh", "C":3800, "C12":2800, "B":7500},
    15:{"name":"Open Data + dữ liệu mở quốc gia", "sector":"Dữ liệu", "C":1500, "C12":1200, "B":3800},
}

SECTOR_COLORS = {
    "Hạ tầng":"#1976D2", "Chính phủ số":"#7B1FA2", "Y tế số":"#E53935",
    "Giáo dục":"#FB8C00", "AI":"#00ACC1", "Tài chính số":"#1A6B3C",
    "Logistics":"#F4511E", "Nông nghiệp":"#7CB342", "Nhân lực":"#D81B60",
    "Bán dẫn":"#3949AB", "An ninh":"#C62828", "Dữ liệu":"#00897B",
}

# Theo đề: hạ tầng 0,85; chính phủ số 0,75; AI/bán dẫn 0,65; còn lại 0,80.
def p_success(i: int) -> float:
    sector = DATA[i]["sector"]
    if sector == "Hạ tầng":
        return 0.85
    if sector == "Chính phủ số":
        return 0.75
    if sector in ["AI", "Bán dẫn"]:
        return 0.65
    return 0.80


def _totals(selected, use_expected=False):
    selected = list(selected)
    benefit = sum(DATA[i]["B"] for i in selected)
    expected = sum(p_success(i) * DATA[i]["B"] for i in selected)
    cost = sum(DATA[i]["C"] for i in selected)
    cost12 = sum(DATA[i]["C12"] for i in selected)
    return {
        "selected": selected,
        "n": len(selected),
        "total_cost": cost,
        "total_c12": cost12,
        "total_benefit": benefit,
        "Z": expected if use_expected else benefit,
        "expected_benefit": expected,
        "ratio": benefit / cost if cost else np.nan,
    }


def _is_feasible(selected, budget=80000, budget12=40000, force_p1p2=False):
    s = set(selected)
    if sum(DATA[i]["C"] for i in s) > budget:
        return False
    if sum(DATA[i]["C12"] for i in s) > budget12:
        return False

    # C3 hoặc kịch bản redundancy.
    if force_p1p2:
        if not (1 in s and 2 in s):
            return False
    else:
        if 1 in s and 2 in s:
            return False

    # C4, C5: tiên quyết nhân lực.
    if 8 in s and 12 not in s:
        return False
    if 13 in s and 12 not in s:
        return False

    # C6: ít nhất một chính phủ số, và P14 bắt buộc.
    if not (4 in s or 5 in s):
        return False
    if 14 not in s:
        return False

    # C7: số lượng dự án.
    if not (7 <= len(s) <= 11):
        return False
    return True


def _solve_exact_by_enumeration(budget=80000, budget12=40000, force_p1p2=False, use_expected=False):
    """Fallback exact: duyệt 2^15 tổ hợp. Dùng được vì bài chỉ có 15 biến nhị phân."""
    best_val = -1e100
    best_cost = 1e100
    best_c12 = 1e100
    best_sel = None
    for mask in range(1 << len(PROJECTS)):
        sel = [i + 1 for i in range(len(PROJECTS)) if (mask >> i) & 1]
        if not _is_feasible(sel, budget=budget, budget12=budget12, force_p1p2=force_p1p2):
            continue
        val = sum((p_success(i) * DATA[i]["B"] if use_expected else DATA[i]["B"]) for i in sel)
        cost = sum(DATA[i]["C"] for i in sel)
        c12 = sum(DATA[i]["C12"] for i in sel)
        # Lexicographic tie-break: max objective -> min total cost -> min early cost.
        if (val > best_val + 1e-9) or (abs(val - best_val) <= 1e-9 and (cost, c12) < (best_cost, best_c12)):
            best_val = val
            best_cost = cost
            best_c12 = c12
            best_sel = sel
    if best_sel is None:
        return {"status": "Infeasible", "method": "Exact enumeration"}
    ans = _totals(best_sel, use_expected=use_expected)
    ans.update({"status": "Optimal", "method": "Exact enumeration"})
    return ans


def _add_constraints(model, y, budget=80000, budget12=40000, force_p1p2=False):
    model += lpSum(DATA[i]["C"] * y[i] for i in PROJECTS) <= budget, "C1_total_budget"
    model += lpSum(DATA[i]["C12"] * y[i] for i in PROJECTS) <= budget12, "C2_year_1_2_budget"
    if force_p1p2:
        model += y[1] == 1, "C3_force_P1"
        model += y[2] == 1, "C3_force_P2"
    else:
        model += y[1] + y[2] <= 1, "C3_data_center_exclusion"
    model += y[8] <= y[12], "C4_AI_requires_training"
    model += y[13] <= y[12], "C5_semiconductor_requires_training"
    model += y[4] + y[5] >= 1, "C6_at_least_one_e_gov"
    model += y[14] >= 1, "C6_cybersecurity_required"
    model += lpSum(y[i] for i in PROJECTS) >= 7, "C7_min_projects"
    model += lpSum(y[i] for i in PROJECTS) <= 11, "C7_max_projects"


def _solve_pulp(budget=80000, budget12=40000, force_p1p2=False, use_expected=False):
    if not HAS_PULP:
        return None

    def objective_expr(y):
        if use_expected:
            return lpSum(p_success(i) * DATA[i]["B"] * y[i] for i in PROJECTS)
        return lpSum(DATA[i]["B"] * y[i] for i in PROJECTS)

    # Stage 1: maximize benefit / expected benefit.
    m = LpProblem("VN_Project_Selection", LpMaximize)
    y = LpVariable.dicts("y", PROJECTS, lowBound=0, upBound=1, cat="Binary")
    obj = objective_expr(y)
    m += obj
    _add_constraints(m, y, budget=budget, budget12=budget12, force_p1p2=force_p1p2)
    m.solve(PULP_CBC_CMD(msg=False))
    if LpStatus[m.status] != "Optimal":
        return {"status": LpStatus[m.status], "method": "PuLP/CBC"}
    z_star = float(value(obj))

    # Stage 2: among optimal solutions, minimize total cost. This removes arbitrary tie choices.
    m2 = LpProblem("VN_Project_Selection_TieBreak", LpMinimize)
    y2 = LpVariable.dicts("y", PROJECTS, lowBound=0, upBound=1, cat="Binary")
    obj2 = objective_expr(y2)
    m2 += lpSum(DATA[i]["C"] * y2[i] for i in PROJECTS)
    _add_constraints(m2, y2, budget=budget, budget12=budget12, force_p1p2=force_p1p2)
    m2 += obj2 >= z_star - 1e-6, "keep_primary_optimum"
    m2.solve(PULP_CBC_CMD(msg=False))
    if LpStatus[m2.status] != "Optimal":
        return {"status": LpStatus[m2.status], "method": "PuLP/CBC"}

    selected = [i for i in PROJECTS if y2[i].value() is not None and y2[i].value() > 0.5]
    ans = _totals(selected, use_expected=use_expected)
    ans.update({"status": "Optimal", "method": "PuLP/CBC", "Z": z_star})
    return ans


@st.cache_data
def solve_mip(budget=80000, budget12=40000, force_p1p2=False, use_expected=False):
    ans = _solve_pulp(budget=budget, budget12=budget12, force_p1p2=force_p1p2, use_expected=use_expected)
    if ans is None or ans.get("status") != "Optimal":
        ans = _solve_exact_by_enumeration(budget=budget, budget12=budget12, force_p1p2=force_p1p2, use_expected=use_expected)
    return ans


def _df_projects(selected=None):
    selected = set(selected or [])
    rows = []
    for i in PROJECTS:
        d = DATA[i]
        rows.append({
            "Chọn": "✓" if i in selected else "",
            "Mã": f"P{i}",
            "Tên dự án": d["name"],
            "Lĩnh vực": d["sector"],
            "Chi phí": d["C"],
            "NPV": d["B"],
            "Năm 1-2": d["C12"],
            "B/C": round(d["B"] / d["C"], 3),
            "pᵢ": p_success(i),
            "pᵢ·NPV": round(p_success(i) * d["B"], 1),
        })
    return pd.DataFrame(rows)


def _selected_table(res):
    rows = []
    for i in res.get("selected", []):
        d = DATA[i]
        rows.append({
            "Mã": f"P{i}",
            "Tên dự án": d["name"],
            "Lĩnh vực": d["sector"],
            "Chi phí": d["C"],
            "NPV": d["B"],
            "Năm 1-2": d["C12"],
            "B/C": round(d["B"] / d["C"], 3),
            "pᵢ": p_success(i),
            "pᵢ·NPV": round(p_success(i) * d["B"], 1),
        })
    return pd.DataFrame(rows)


def _kpi_card(col, label, value, sub, color):
    with col:
        st.markdown(f"""
        <div style="background:#fff;border:1.5px solid #E0EBE4;border-radius:12px;
                    padding:0.9rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);text-align:center;">
            <div style="font-size:0.72rem;font-weight:700;color:#6B8A7A;text-transform:uppercase;margin-bottom:5px;">{label}</div>
            <div style="font-size:1.45rem;font-weight:900;color:{color};line-height:1;">{value}</div>
            <div style="font-size:0.72rem;color:#8AA898;margin-top:4px;">{sub}</div>
        </div>""", unsafe_allow_html=True)


def _compare_sets(old, new):
    old_s = set(old.get("selected", [])); new_s = set(new.get("selected", []))
    added = sorted(new_s - old_s)
    removed = sorted(old_s - new_s)
    return added, removed


def _project_badges(projects, bg="#E8F5E9", border=C1):
    if not projects:
        st.markdown("<div style='color:#6B8A7A;font-size:0.86rem;'>Không có thay đổi.</div>", unsafe_allow_html=True)
        return
    for i in projects:
        st.markdown(f"""
        <div style="background:{bg};border-left:3px solid {border};border-radius:6px;
                    padding:8px 12px;margin:5px 0;font-size:0.85rem;">
            <b>P{i}</b> — {DATA[i]['name']} · C={DATA[i]['C']:,} · NPV={DATA[i]['B']:,}
        </div>""", unsafe_allow_html=True)


def render():
    bai_header(
        so="5",
        ten="Quy hoạch nguyên hỗn hợp lựa chọn dự án chuyển đổi số",
        mo_ta="15 dự án, biến nhị phân yᵢ ∈ {0,1}, ràng buộc loại trừ, tiên quyết, bắt buộc và ngân sách đa năm",
        cap_do="TB",
        tools=["PuLP", "CBC solver", "Exact enumeration fallback", "plotly"],
        thoi_luong="1.5 tuần",
    )

    base = solve_mip(budget=80000, budget12=40000)
    # Theo nghĩa chặt của câu 5.4.2: chỉ nới ngân sách tổng, giữ nguyên C2 = 40.000.
    b100_strict = solve_mip(budget=100000, budget12=40000)
    # Kịch bản phụ: nếu ngân sách năm 1-2 cũng được nới tương ứng lên 50.000.
    b100_scaled = solve_mip(budget=100000, budget12=50000)
    force12 = solve_mip(budget=80000, budget12=40000, force_p1p2=True)
    expected = solve_mip(budget=80000, budget12=40000, use_expected=True)

    info_box(
        "Mô hình: <b>max Σ Bᵢ·yᵢ</b>, với <b>yᵢ ∈ {0,1}</b>. "
        "Ràng buộc gồm: ngân sách 5 năm, ngân sách năm 1–2, loại trừ P1/P2, tiên quyết P8/P13 ⇒ P12, "
        "ít nhất một dự án chính phủ số, bắt buộc P14, và 7 ≤ số dự án ≤ 11.<br>"
        f"Trạng thái solver: <b>{base['status']}</b> bằng <b>{base['method']}</b>.",
        bg="#E8F5E9", border=C1, icon="📐",
    )

    section_title("Danh mục 15 dự án ứng cử", "📋")
    st.dataframe(_df_projects(), use_container_width=True, hide_index=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 5.4.1
    # ════════════════════════════════════════════════════════
    section_title("Câu 5.4.1 — Nghiệm tối ưu MIP với ngân sách 80.000 tỷ", "🎯")

    cols = st.columns(5, gap="small")
    _kpi_card(cols[0], "Z* NPV", f"{base['total_benefit']:,.0f}", "tỷ VND", C1)
    _kpi_card(cols[1], "Tổng chi phí", f"{base['total_cost']:,.0f}", f"{base['total_cost']/80000*100:.1f}% NS", CBLUE)
    _kpi_card(cols[2], "Năm 1–2", f"{base['total_c12']:,.0f}", "/40.000 tỷ", CORANGE)
    _kpi_card(cols[3], "Số dự án", f"{base['n']}/15", "trong [7,11]", CPURPLE)
    _kpi_card(cols[4], "NPV/Chi phí", f"{base['ratio']:.3f}", "NPV biên", C2)

    st.markdown("**Dự án được chọn:**")
    st.dataframe(_selected_table(base), use_container_width=True, hide_index=True)

    info_box(
        "Có hai nghiệm cùng đạt NPV 115.400 trong bài toán gốc. Phiên bản này dùng tie-break hợp lý: "
        "nếu NPV bằng nhau thì chọn danh mục có chi phí thấp hơn. Vì vậy nghiệm báo cáo là "
        "<b>P2, P4, P6, P7, P8, P9, P12, P14, P15</b>, chi phí 59.600 tỷ và NPV/chi phí ≈ 1,936.",
        bg="#F1F8F2", border=C2, icon="✅",
    )

    col_a, col_b = st.columns([2, 3], gap="large")
    with col_a:
        sector_cost = {}
        for i in base["selected"]:
            sector = DATA[i]["sector"]
            sector_cost[sector] = sector_cost.get(sector, 0) + DATA[i]["C"]
        fig_pie = go.Figure(go.Pie(
            labels=list(sector_cost.keys()),
            values=list(sector_cost.values()),
            hole=0.45,
            marker_colors=[SECTOR_COLORS.get(s, "#888") for s in sector_cost],
            textinfo="percent",
        ))
        fig_pie.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0), font=CF,
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
    with col_b:
        sel_sorted = sorted(base["selected"], key=lambda i: DATA[i]["B"], reverse=True)
        fig_bar = go.Figure(go.Bar(
            x=[f"P{i}" for i in sel_sorted],
            y=[DATA[i]["B"] for i in sel_sorted],
            marker_color=[SECTOR_COLORS.get(DATA[i]["sector"], "#888") for i in sel_sorted],
            text=[f"{DATA[i]['B']:,}" for i in sel_sorted],
            textposition="outside",
        ))
        fig_bar.update_layout(height=270, margin=dict(l=0, r=0, t=20, b=0), font=CF,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, yaxis=dict(showgrid=True, gridcolor="#F0F4F0"))
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 5.4.2
    # ════════════════════════════════════════════════════════
    section_title("Câu 5.4.2 — Nới ngân sách tổng lên 100.000 tỷ", "💰")

    cols = st.columns(4, gap="small")
    _kpi_card(cols[0], "Gốc 80k", f"{base['total_benefit']:,.0f}", "NPV", "#8AA898")
    _kpi_card(cols[1], "100k, C2 giữ 40k", f"{b100_strict['total_benefit']:,.0f}", "theo nghĩa chặt của đề", C1)
    _kpi_card(cols[2], "100k, C2 = 50k", f"{b100_scaled['total_benefit']:,.0f}", "kịch bản phụ", CORANGE)
    _kpi_card(cols[3], "Nút thắt", "C2", "ngân sách năm 1–2", CRED)

    info_box(
        "Nếu chỉ nới <b>ngân sách tổng 5 năm</b> từ 80.000 lên 100.000 nhưng giữ nguyên ngân sách năm 1–2 ≤ 40.000, "
        "nghiệm <b>không đổi</b>. Lý do: ràng buộc năm 1–2 đang gần chạm trần 39.800/40.000, nên tiền tăng thêm "
        "ở ngân sách tổng không sử dụng được nếu không nới C2. Nếu giảng viên muốn hiểu là nới cả ngân sách năm 1–2 "
        "theo tỷ lệ lên 50.000, khi đó NPV tăng lên 142.500.",
        bg="#E3F2FD", border=CBLUE, icon="🔍",
    )

    added, removed = _compare_sets(base, b100_scaled)
    c_add, c_rm = st.columns(2, gap="large")
    with c_add:
        st.markdown("**Kịch bản phụ C2 = 50.000 — dự án được thêm:**")
        _project_badges(added, bg="#E8F5E9", border=C1)
    with c_rm:
        st.markdown("**Kịch bản phụ C2 = 50.000 — dự án bị thay thế:**")
        _project_badges(removed, bg="#FFF3E0", border=CORANGE)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 5.4.3
    # ════════════════════════════════════════════════════════
    section_title("Câu 5.4.3 — Bắt buộc có cả P1 và P2 để redundancy", "⚖️")

    delta = base["total_benefit"] - force12["total_benefit"] if force12["status"] == "Optimal" else np.nan
    cols = st.columns(4, gap="small")
    _kpi_card(cols[0], "Trạng thái", force12["status"], force12["method"], C1 if force12["status"] == "Optimal" else CRED)
    _kpi_card(cols[1], "Z* mới", f"{force12['total_benefit']:,.0f}" if force12["status"] == "Optimal" else "—", "tỷ VND", CORANGE)
    _kpi_card(cols[2], "Chi phí", f"{force12['total_cost']:,.0f}" if force12["status"] == "Optimal" else "—", "tỷ VND", CBLUE)
    _kpi_card(cols[3], "Giảm NPV", f"{delta:,.0f}" if force12["status"] == "Optimal" else "—", "tỷ VND", CRED)

    st.dataframe(_selected_table(force12), use_container_width=True, hide_index=True)
    info_box(
        "Kịch bản redundancy vẫn <b>khả thi</b>. Ta thay C3 loại trừ P1/P2 bằng điều kiện y₁ = 1 và y₂ = 1. "
        "NPV giảm từ 115.400 xuống 113.300, tức chi phí cơ hội khoảng <b>2.100 tỷ VND</b>. "
        "Về chính sách, đây là đánh đổi giữa hiệu quả tài chính và khả năng dự phòng hệ thống dữ liệu quốc gia.",
        bg="#FFF8E1", border=CORANGE, icon="💡",
    )

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU 5.4.4
    # ════════════════════════════════════════════════════════
    section_title("Câu 5.4.4 — Mở rộng rủi ro: tối đa hóa lợi ích kỳ vọng E[Z]", "🎲")

    cols = st.columns(5, gap="small")
    _kpi_card(cols[0], "E[Z]*", f"{expected['expected_benefit']:,.0f}", "tỷ VND", CPURPLE)
    _kpi_card(cols[1], "NPV danh nghĩa", f"{expected['total_benefit']:,.0f}", "của danh mục kỳ vọng", C1)
    _kpi_card(cols[2], "Tổng chi phí", f"{expected['total_cost']:,.0f}", "tỷ VND", CBLUE)
    _kpi_card(cols[3], "Năm 1–2", f"{expected['total_c12']:,.0f}", "/40.000 tỷ", CORANGE)
    _kpi_card(cols[4], "Số dự án", f"{expected['n']}/15", "trong [7,11]", CPURPLE)

    st.dataframe(_selected_table(expected), use_container_width=True, hide_index=True)

    changed = sorted(set(base["selected"]).symmetric_difference(set(expected["selected"])))
    info_box(
        "Khi đưa rủi ro tiến độ vào mục tiêu, mô hình bớt ưu tiên các dự án AI có p thấp. "
        "Danh mục kỳ vọng chọn <b>P2, P3, P5, P6, P7, P12, P14, P15</b>, đạt E[Z] = 91.285. "
        f"Các dự án thay đổi so với cơ sở: {', '.join('P'+str(i) for i in changed)}.",
        bg="#F3E5F5", border=CPURPLE, icon="🔍",
    )

    fig_p = go.Figure(go.Bar(
        x=[f"P{i}" for i in PROJECTS],
        y=[p_success(i) * DATA[i]["B"] for i in PROJECTS],
        marker_color=[C1 if i in expected["selected"] else "#D7E3DC" for i in PROJECTS],
        text=[f"{p_success(i)*DATA[i]['B']:,.0f}" for i in PROJECTS],
        textposition="outside",
    ))
    fig_p.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), font=CF,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False, yaxis=dict(showgrid=True, gridcolor="#F0F4F0", title="pᵢ·NPV"),
                        xaxis=dict(showgrid=False))
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CÂU HỎI THẢO LUẬN
    # ════════════════════════════════════════════════════════
    section_title("Câu hỏi thảo luận chính sách", "💬")

    q_data = [
        (
            "a)",
            "Vì sao mô hình có thể bỏ qua P15 dù B/C rất cao? Đây có phải kết quả mong muốn không?",
            "Về mặt kỹ thuật, bài toán tối đa hóa <b>NPV tuyệt đối</b>, không tối đa hóa B/C. Vì vậy một dự án nhỏ như P15 "
            "có thể bị loại nếu ràng buộc số lượng dự án ≤ 11 làm nó cạnh tranh vị trí với dự án có NPV tuyệt đối lớn hơn. "
            "Trong nghiệm tối ưu của đề hiện tại, P15 vẫn được chọn vì chi phí rất thấp và giúp lấp danh mục trong khi không phá ngân sách năm 1–2. "
            "Nếu ở biến thể nào đó P15 bị bỏ, đó là tín hiệu chính sách cần xem lại hàm mục tiêu: dữ liệu mở có giá trị nền tảng, "
            "nên có thể thêm ràng buộc bắt buộc hoặc cộng điểm chiến lược riêng.",
        ),
        (
            "b)",
            "Ràng buộc bắt buộc P14 có làm giảm Z* không? Có hợp lý không?",
            "Trong nghiệm hiện tại, P14 được chọn ngay cả khi không cần bắt buộc vì có chi phí 3.800, NPV 7.500 và B/C ≈ 1,974. "
            "Do đó ràng buộc bắt buộc P14 <b>không làm giảm Z*</b> trong bộ dữ liệu này. Về chính sách, bắt buộc an ninh mạng là hợp lý "
            "vì các dự án dữ liệu, định danh, y tế, giáo dục và AI đều làm tăng bề mặt rủi ro; thiếu SOC quốc gia có thể khiến lợi ích số hóa "
            "bị vô hiệu hóa bởi rủi ro vận hành và niềm tin công chúng.",
        ),
        (
            "c)",
            "Mô hình hóa cộng hưởng P8 và P13 như thế nào?",
            "Thêm biến nhị phân phụ <b>z₈,₁₃</b> biểu diễn việc chọn đồng thời P8 và P13. Linear hóa bằng ba ràng buộc: "
            "z₈,₁₃ ≤ y₈, z₈,₁₃ ≤ y₁₃, z₈,₁₃ ≥ y₈ + y₁₃ − 1. Sau đó sửa hàm mục tiêu thành "
            "max ΣBᵢyᵢ + S·z₈,₁₃, trong đó S là lợi ích cộng hưởng. Cách này vẫn giữ mô hình là MIP tuyến tính.",
        ),
    ]
    for code, q, ans in q_data:
        with st.expander(f"{code} {q}", expanded=False):
            info_box(ans, bg="#F1F8F2", border=C2, icon="✅")

    end_padding()
