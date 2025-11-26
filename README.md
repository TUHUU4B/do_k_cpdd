# Độ chặt cấp phối đá dăm – AASHTO T191

Ứng dụng Streamlit hỗ trợ kỹ sư hiện trường xác định khối lượng thể tích khô và độ chặt của cấp phối đá dăm theo phương pháp rót cát (Sand Cone - AASHTO T191). Ứng dụng cho phép:

- Nhập hoặc hiệu chuẩn khối lượng riêng của cát chuẩn.
- Tính hoặc nhập trực tiếp độ ẩm của mẫu.
- Nhập số liệu cân ngoài hiện trường và xem nhanh các kết quả như thể tích hố, khối lượng thể tích ẩm, khối lượng thể tích khô và phần trăm độ chặt so với yêu cầu.
- Xuất bảng kết quả trực tiếp trên giao diện.

## Yêu cầu hệ thống

- Python 3.10+
- Các thư viện trong `requirements.txt`

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở trên trình duyệt tại địa chỉ `http://localhost:8501`.

## Cấu trúc dữ liệu chính

- `SandCalibration`: xử lý phần hiệu chuẩn cát.
- `compute_moisture_content()`: tính độ ẩm từ số liệu sấy hoặc nhập trực tiếp.
- `compute_field_results()`: tính toán khối lượng thể tích ẩm/khô, thể tích hố.

## Tùy biến

- Thay logo bằng cách cập nhật `logo.png` trong thư mục gốc.
- Điều chỉnh nội dung, ngôn ngữ hiển thị trong `app.py` tùy nhu cầu dự án.

## Góp ý

Liên hệ: **MR Tuấn – 0946 135 156** (Công ty Tứ Hữu).

