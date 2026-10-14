# Checklist nghiệm thu Lab Day 10

## Trước khi nộp

- [x] `REPORT.md` ghi họ tên **Nguyễn Hoàng Duy**, tên hiển thị Colab **Stevens Lauren**, `dataset_version` **d10-21e4070301**, bảng E0–E4 và đủ 10 câu trả lời.
- [x] Không còn chuỗi `<điền` trong `REPORT.md`.
- [x] Báo cáo nhúng 6 ảnh trở lên: ba cặp E0/E3 trên cùng các frame test; thư mục `report/figures/` có 12 ảnh.
- [x] `report/evidence/` có E0–E4, gồm hai JSON E4: kết quả test rank 64 và kết quả chọn rank trên validation.
- [x] `data/splits/meta.json` khớp dataset version và số ảnh/box của ba split.
- [x] `python scripts/check_submission.py` chạy với UTF-8: **0 FAIL, 0 WARN**.
- [x] `git rev-list --count HEAD`: **7**, đạt ngưỡng tối thiểu **5**.
- [x] `git ls-files` không có `data/`, `work/`, `.env`, `*.safetensors` hoặc `*.pt`; `.env.example` là mẫu cấu hình và không chứa secrets.
- [x] Không có thay đổi trong `scripts/`, `tests/` hoặc `configs/`.

## Phát hành

- [x] Commit và push lên `origin/main` — commit `fdc8580`.
- [x] Xác nhận repository GitHub ở trạng thái Public và nộp URL qua form Day 10 trên VLearn (đánh giá 3/5 sao).
- [ ] Giữ repository Public đến khi điểm chính thức được công bố.

Repository cần tiếp tục ở trạng thái Public cho đến khi điểm chính thức được công bố.
