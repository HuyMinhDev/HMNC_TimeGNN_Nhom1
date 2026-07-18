# TimeGNN - Oil Market Forecasting

## Giới thiệu

Đây là dự án mô phỏng thuật toán **TimeGNN** (Time-series Graph Neural Network) dùng để dự báo giá dầu tại thị trường Việt Nam và quốc tế. Thuật toán kết hợp **Graph Neural Network (GNN)** với **GRU** để học cả mối quan hệ giữa các loại dầu và xu hướng theo thời gian.

---

## Cấu trúc File

### 1. `TimeGNN_OilMarket.ipynb`
**File chính** - Jupyter Notebook chứa toàn bộ code và pipeline huấn luyện.

**Các bước trong notebook:**

| Bước | Nội dung | Mô tả |
|------|----------|-------|
| 1 | **Load & Tiền xử lý dữ liệu** | Đọc file CSV, chuẩn hóa dữ liệu (MinMaxScaler) |
| 2 | **Xây dựng đồ thị tương quan** | Tạo ma trận kề dựa trên hệ số tương quan giữa các biến |
| 3 | **Xây dựng mô hình TimeGNN** | Kết hợp GCN (Graph Convolutional Network) + GRU |
| 4 | **Huấn luyện với Sliding Window** | Sử dụng cửa sổ trượt để tạo sequences cho training |
| 5 | **Đánh giá** | Tính các chỉ số MAE, RMSE, MAPE, R² |

**Thư viện sử dụng:**
- `torch`, `torch-geometric` - Deep learning & GNN
- `pandas`, `numpy` - Xử lý dữ liệu
- `scikit-learn` - Tiền xử lý & đánh giá
- `matplotlib`, `seaborn` - Trực quan hóa

---

### 2. `oil_market_GNN.csv`
**File dữ liệu** - Bộ dữ liệu chuỗi thời gian về giá dầu.

**Thông tin dữ liệu:**
- **Thời gian:** 2008-05-02 → 2026-03-18 (~18 năm)
- **Số dòng:** 4,610 bản ghi (ngày giao dịch)
- **Số cột:** 16

**Các cột dữ liệu:**

| Cột | Mô tả | Ví dụ giá trị |
|-----|-------|---------------|
| `Ngày` | Ngày giao dịch | 2008-05-02 |
| `MG95` | Giá xăng RON 95 (thị trường VN) | 115.1 |
| `MG92` | Giá xăng RON 92 (thị trường VN) | 114.37 |
| `DO 0.001%` | Giá dầu diesel (VN) | 150.34 |
| `DO 0.05%` | Giá dầu diesel loại khác (VN) | 141.14 |
| `BRT DTD` | Giá dầu Brent Dubai (thị trường quốc tế) | 111.89 |
| `BRT KH` | Giá dầu Brent Kuwait (thị trường quốc tế) | 113.73 |
| `WTI` | West Texas Intermediate (giá dầu tham chiếu Mỹ) | 116.33 |
| `USD_Index` | Chỉ số đồng USD | 87.23 |
| `GPRD` | Chỉ số giá tiêu dùng / lạm phát | 57.53 |
| `NgayTrongTuan` | Ngày trong tuần (1-7) | 5 |
| `ThangTrongNam` | Tháng trong năm (1-12) | 5 |
| `QuyTrongNam` | Quý trong năm (1-4) | 2 |
| `Nam` | Năm | 2008 |
| `NgayLe` | Cờ ngày lễ (0/1) | 0 |
| `SuKienDacBiet` | Cờ sự kiện đặc biệt (0/1) | 0 |

**Lưu ý:** Dữ liệu sử dụng dấu `,` làm decimal separator (định dạng châu Âu).

---

### 3. `TimeGNN_OilMarket.pptx`
**File trình chiếu** - PowerPoint presentation giới thiệu tổng quan dự án.

Nội dung bao gồm:
- Giới thiệu bài toán dự báo giá dầu
- Mô hình TimeGNN (kiến trúc GCN + GRU)
- Pipeline huấn luyện
- Kết quả và đánh giá
- Kết luận

---

### 4. `.git/`
**Thư mục Git** - Quản lý version control cho dự án.

---

## Kiến trúc mô hình TimeGNN

```
┌─────────────────────────────────────────────────────────────┐
│                    TimeGNN Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Node Input  │───▶│   GCN Layer  │───▶│   GCN Layer  │  │
│  │  (Features)  │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                       │            │
│         ▼                                       ▼            │
│  ┌──────────────┐                      ┌──────────────┐       │
│  │  Adjacency  │                      │  GRU Layer   │       │
│  │   Matrix    │                      │  (Temporal)  │       │
│  └──────────────┘                      └──────────────┘       │
│                                              │                │
│                                              ▼                │
│                                    ┌──────────────┐           │
│                                    │   Output     │           │
│                                    │   Layer      │           │
│                                    └──────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Giải thích:**
- **GCN (Graph Convolutional Network):** Học mối quan hệ giữa các biến thông qua đồ thị tương quan
- **GRU (Gated Recurrent Unit):** Học xu hướng và chuỗi thời gian

---

## Cách chạy

### 1. Trên Google Colab (Khuyến nghị)
1. Upload file `TimeGNN_OilMarket.ipynb` lên Google Colab
2. Upload file `oil_market_GNN.csv` vào thư mục `/content/`
3. Chạy tất cả các cells theo thứ tự

### 2. Trên máy local
```bash
# Cài đặt thư viện
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn

# Chạy notebook
jupyter notebook TimeGNN_OilMarket.ipynb
```

**Lưu ý:** Cần GPU để huấn luyện hiệu quả (code tự động dùng CUDA nếu có).

---

## Đánh giá mô hình

Các chỉ số được sử dụng:
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
- **RMSE** (Root Mean Square Error): Sai số bình phương trung bình gốc
- **MAPE** (Mean Absolute Percentage Error): Phần trăm sai số tuyệt đối trung bình
- **R²** (R-squared): Hệ số xác định

---

## Tác giả

Dự án được phát triển cho mục đích nghiên cứu và học tập về ứng dụng Graph Neural Network trong dự báo chuỗi thời gian.
