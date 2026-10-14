# Báo cáo lỗi và lưu ý kỹ thuật — Lab Day 10

## Môi trường và dữ liệu

- Colab chạy bằng GPU A100 trên tài khoản hiển thị **Stevens Lauren**; Google Drive được mount để lưu checkpoint và minh chứng.
- Model Qwen3-VL-4B được tải công khai; lab không cần `HF_TOKEN`. Không nhập token hay credential vào notebook.
- Split được nhóm theo phiên quay để tránh các frame liền kề rơi vào cả train và test. Metadata và evidence dùng chung `dataset_version` **d10-21e4070301**.
- Nếu chạy lại trên Colab, cần kiểm tra đúng tài khoản, runtime A100, Drive đã mount và checkpoint có sẵn trước khi bật các cell huấn luyện dài.

## Lỗi lượng tử hóa projector khi nạp checkpoint

Trong môi trường Transformers 5.19, bitsandbytes lượng tử hóa các merger của phần vision. Tensor dense trong checkpoint projector sau đó không khớp hình dạng tensor đóng gói của `Params4bit` khi restore, gây lỗi copy weight (ví dụ dense `[4096, 4096]` và dạng packed `[8388608, 1]`).

Đã cấu hình bỏ `model.visual.merger`, `model.visual.deepstack_merger_list` và `lm_head` khỏi lượng tử hóa; các projector cần train giữ dạng dense. Hàm nạp checkpoint kiểm tra key và shape trước khi copy, đồng thời xác nhận merger không bị chuyển thành `Linear4bit`. LLM vẫn chạy 4-bit với LoRA. Projector trainable được đưa về float32 để cập nhật ổn định; khi suy luận có thể chuyển về dtype của model.

## Sai số so sánh reward

Ô kiểm tra E5 từng so sánh trực tiếp số thực `0.6000000000000001 == 0.6`, khiến assertion thất bại dù reward đúng về mặt số học. Đã đổi sang `math.isclose` với dung sai `1e-9`; kiểm tra perfect match, JSON lỗi, sai lớp, box thừa và hai tập rỗng đều qua. GRPO vẫn là phần tuỳ chọn, không được khởi chạy.

## Cảnh báo suy luận

Transformers in cảnh báo tham số processor cần nằm trong `processor_kwargs`. Cảnh báo không làm dừng sinh dự đoán: E4 đã hoàn tất 311/311 ảnh, `parse_valid_rate` 1.0 và không thiếu prediction. Khi nâng phiên bản thư viện, nên xác minh API processor rồi cập nhật cách truyền tham số riêng; không nên bỏ qua exception thật.

## Đọc kết quả E4

Rank 4 và rank 64 đều hoàn tất 300 bước. Rank 64 được chọn theo micro-F1 **0.3763** trên cùng 40 ảnh validation; test chỉ chạy một lần cho rank được chọn. E4 rank 64 đạt test F1 **0.4322**, CI95 **[0.3861, 0.4746]**. Điểm nhỉnh hơn E3 trong lần chạy này nhưng CI hai setup chồng lấp; chưa thể khẳng định cải thiện có ý nghĩa thống kê nếu chưa bootstrap ghép cặp.

## Nghiệm thu và bản Colab tạm

Chạy nghiệm thu ở thư mục gốc của workspace, nơi có E0–E4, metadata và báo cáo đầy đủ. Một checkout tạm bên trong Colab có thể thiếu evidence E0–E3, bản `REPORT.md` đã điền hoặc lịch sử Git đầy đủ; checker tại đó không thay thế kết quả checker của repository workspace. Không sửa `scripts/`, `tests/` hay `configs/` để làm cho checker qua giả tạo.
