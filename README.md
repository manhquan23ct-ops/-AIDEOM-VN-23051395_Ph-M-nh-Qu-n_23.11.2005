# 🇻🇳 AIDEOM-VN — AI-Driven Economic Optimization Model for Vietnam

> Đồ án cuối kỳ · Môn: Các mô hình ra quyết định · Đại học Kinh tế - ĐHQGHN

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Giới thiệu

**AIDEOM-VN** là dashboard tương tác tích hợp 12 mô hình tối ưu hóa và học máy ứng dụng vào dữ liệu kinh tế vĩ mô Việt Nam 2020–2025. Hệ thống gồm 6 module chính (M1–M6) cho phép phân tích, tối ưu và mô phỏng các kịch bản chính sách.

---

## Cấu trúc dự án

```
AIDEOM-VN/
├── app.py                  # Entry point — routing & sidebar
├── data_loader.py          # Cached CSV loaders
├── utils.py                # Design system, shared components
├── requirements.txt        # Dependencies
├── data/
│   ├── vietnam_macro_2020_2025.csv     # GDP, CPI, FDI, xuất nhập khẩu
│   ├── vietnam_sectors_2024.csv        # 10 ngành kinh tế
│   └── vietnam_regions_2024.csv        # 6 vùng kinh tế
└── pages/
    ├── home.py             # Trang chủ
    ├── bai1.py             # Bài 1 — Cobb-Douglas TFP
    ├── bai2.py             # Bài 2 — LP ngân sách số hóa
    ├── bai3.py             # Bài 3 — Prioritization 10 ngành
    ├── bai4.py             # Bài 4 — LP ngành-vùng
    ├── bai5.py             # Bài 5 — MIP 15 dự án
    ├── bai6.py             # Bài 6 — TOPSIS 6 vùng
    ├── bai7.py             # Bài 7 — NSGA-II Pareto
    ├── bai8.py             # Bài 8 — Mô hình động 2026-2035
    ├── bai9.py             # Bài 9 — Lao động & AI
    ├── bai10.py            # Bài 10 — Stochastic Programming
    ├── bai11.py            # Bài 11 — Q-learning RL
    └── bai12.py            # Bài 12 — AIDEOM tích hợp
```

---

## Cài đặt & Chạy

### Yêu cầu hệ thống
- Python 3.10+
- pip hoặc conda

### Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/AIDEOM-VN.git
cd AIDEOM-VN

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Chạy app
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

---

## Dữ liệu

| File | Mô tả | Nguồn |
|------|--------|-------|
| `vietnam_macro_2020_2025.csv` | 19 chỉ số kinh tế vĩ mô theo năm | GSO, WB |
| `vietnam_sectors_2024.csv` | 10 ngành: GDP share, tăng trưởng, FDI, AI readiness | MoST, GII |
| `vietnam_regions_2024.csv` | 6 vùng: GRDP, dân số, xuất khẩu, digital index | NSO, MIC |

---

## 12 Bài toán

| # | Tên | Phương pháp | Độ khó |
|---|-----|-------------|--------|
| 1 | Cobb-Douglas + TFP | Hồi quy, dự báo | DỄ |
| 2 | LP ngân sách số hóa | Linear Programming | DỄ |
| 3 | Prioritization 10 ngành | AHP / Scoring | DỄ |
| 4 | LP ngành-vùng | LP đa mục tiêu | TB |
| 5 | MIP 15 dự án | Mixed Integer Programming | TB |
| 6 | TOPSIS 6 vùng | MCDM / Entropy Weight | TB |
| 7 | NSGA-II Pareto | Genetic Algorithm | KHÁ KHÓ |
| 8 | Mô hình động 2026–2035 | Dynamic Programming | KHÁ KHÓ |
| 9 | Lao động & AI | Markov Chain, Simulation | KHÁ KHÓ |
| 10 | Stochastic Programming | SP / Monte Carlo | KHÓ |
| 11 | Q-learning RL | Reinforcement Learning | KHÓ |
| 12 | AIDEOM tích hợp | M1–M6 + 5 kịch bản | KHÓ |

---

## Công nghệ

- **Framework**: Streamlit
- **Optimization**: PuLP/CBC, CVXPY, Pyomo, Pymoo (NSGA-II)
- **ML/RL**: Stable-Baselines3, Gymnasium
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data**: Pandas, NumPy, SciPy

---

## Sinh viên thực hiện

> Phí Mạnh Quân — 23051395 — Lớp QH2023E QTKD 1

---

*Dữ liệu sử dụng cho mục đích học thuật. Nguồn: Tổng cục Thống kê (GSO), Bộ KH&CN, World Bank, GII 2025.*
"# -AIDEOM-VN-23051395_Ph-M-nh-Qu-n_23.11.2005" 
