## Phase Hiện Tại

**Phase 0 — Lập kế hoạch & Xác định Phạm vi Dự án**

---

## Mục tiêu

Thiết lập kế hoạch dự án, định hướng sinh học và lộ trình công nghệ cho **NifPredict** (`nif-predict`) — một nền tảng sinh học tính toán nguồn mở tích hợp học máy.

Nền tảng này nhằm dự đoán khả năng cố định đạm (*diazotrophy*) của các chủng vi khuẩn dựa trên dữ liệu bộ gen và dữ liệu sinh thái/phân loại học.

---

## Tầm quan trọng của Bước này

Xây dựng dự án từ con số 0 đòi hỏi phải xác định rõ ranh giới sinh học, giả định kỹ thuật và các giới hạn phạm vi. Nếu không có nền tảng này, các dự án sinh học tính toán rất dễ gặp hiện tượng rò rỉ dữ liệu (target leakage), kiến trúc đường ống (pipeline) dễ gãy, hoặc mô hình đưa ra kết quả không có giá trị sinh học.

Bằng cách chốt phương án dữ liệu, stack công nghệ và tiêu chuẩn sinh học ngay từ đầu, mọi module, script và đặc trưng (feature) sau này sẽ được tối ưu chính xác cho mục tiêu nghiên cứu.

---

## Kế hoạch Chi tiết

### 1. Câu hỏi Nghiên cứu & Cơ sở Sinh học

* **Câu hỏi chính:** Làm thế nào để dự đoán chính xác một bộ gen vi khuẩn chưa rõ đặc tính có khả năng cố định đạm (*diazotrophy*) hay không bằng cách kết hợp hồ sơ gen đánh dấu (hệ gen *nif/fix/anf/vnf*), đặc trưng chức năng toàn bộ gen và dữ liệu phân loại/sinh thái?
* **Xác định Nhãn Target:**
* **Nhóm Dương tính (Class $1$):** Các vi khuẩn cố định đạm đã xác minh hoặc có độ tin cậy cao, sở hữu nhóm gen mã hóa phức hợp enzyme nitrogenase (*nifHDK*, hoặc các enzyme thay thế *vnfHDK*, *anfHDK*) cùng hệ thống hỗ trợ sinh tổng hợp/vận chuyển electron (*nifENB*, *fixABCX*).
* **Nhóm Âm tính (Class $0$):** Các vi khuẩn không cố định đạm thuộc nhiều ngành khác nhau, bao gồm cả những loài có quan hệ họ hàng gần với nhóm cố định đạm (tránh trường hợp mô hình chỉ học vị trí phân loại thay vì chức năng sinh học).


* **Giả định Sinh học cốt lõi:**
* Khả năng cố định đạm bắt buộc phải có sự xuất hiện của bộ ba gen xúc tác nòng nồng (*nifH*, *nifD*, *nifK* hoặc biến thể chức năng).
* Bối cảnh bộ gen và các gen phụ trợ (*nifE*, *nifN*, *nifB*, cụm *fix*) cung cấp tín hiệu bổ sung mạnh mẽ giúp phân biệt chủng cố định đạm thực sự với giả gen (pseudogenes) hoặc các đoạn gen chuyển ngang (HGT) không hoàn chỉnh.



### 2. Rủi ro & Giới hạn

* **Chuyển gen ngang (HGT):** Cụm gen *nif* có tính linh động cao. Một bộ gen có thể chứa đoạn gen *nif* rải rác nhưng thiếu chức năng.
* **Độ lệch phân loại (Taxonomic Bias):** Các chủng cố định đạm mẫu (như *Rhizobium*, *Azotobacter*, *Klebsiella*) xuất hiện quá nhiều có thể gây mất cân bằng dữ liệu và làm mô hình bị overfit theo nhóm loài.
* **Chất lượng Genbank:** Bộ gen từ môi trường (MAGs) có thể không hoàn chỉnh hoặc lẫn tạp chất.

### 3. Công nghệ Sử dụng

* **Nguồn Dữ liệu:**
* *Genomics:* NCBI Assembly / Datasets (RefSeq/GenBank), GTDB (Genome Taxonomy Database).
* *Metadata & Phenotypes:* BacDive, tài liệu khoa học.


* **Tech Stack:**
* **Ngôn ngữ:** Python 3.10+
* **Khoa học dữ liệu & ML:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`
* **Công cụ Sinh tin:** `HMMER` (qua `pyhmmer` hoặc subprocess), `pyrodigal` (dự đoán gen), `NCBI Datasets CLI`
* **Kỹ thuật phần mềm:** `pydantic`, `dataclasses`, `pathlib`, `logging`, `argparse`



---

## Lộ trình Phát triển Dự án

```
PHASE 0: Lập kế hoạch & Định hướng Sinh học  [ĐANG THỰC HIỆN]
   │
   ▼
PHASE 1: Kiến trúc Phần mềm & Thiết lập Môi trường
   │
   ▼
PHASE 2: Thu thập Dữ liệu Tự động (NCBI / GTDB / BacDive)
   │
   ▼
PHASE 3: Xử lý Dữ liệu & Chuẩn hóa Metadata
   │
   ▼
PHASE 4: Xây dựng Tập dữ liệu Chuẩn (Gold-Standard Dataset)
   │
   ▼
PHASE 5: Chú giải Sinh học (Annotation với HMM nif/fix/anf/vnf)
   │
   ▼
PHASE 6: Trích xuất Đặc trưng (Feature Engineering Matrix)
   │
   ▼
PHASE 7: Phân tích Khám phá Dữ liệu (EDA & Kiểm định Chất lượng)
   │
   ▼
PHASE 8: Huấn luyện Mô hình ML & Phân tích SHAP
   │
   ▼
PHASE 9: Đường ống Dự đoán Đầu-Cuối (Prediction Pipeline)
   │
   ▼
PHASE 10: Đóng gói Sản phẩm (CLI, Docker, CI/CD, Document)

```

---

## Kết quả Kỳ vọng

Một bản kế hoạch dự án hoàn chỉnh và thống nhất quy trình thực hiện giữa hai bên.


