# Báo cáo Lab Day 10 — Qwen3-VL-4B

Họ tên: **Nguyễn Hoàng Duy**<br>
Tên hiển thị Colab: **Stevens Lauren**<br>
`dataset_version`: **d10-21e4070301**

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | zero-shot | 0 | — | — | 0.0965 | [0.0711, 0.1254] | 1.170 | 0.1127 | 0.0327 | 0.2490 |
| E1 | LoRA LLM | 33,030,144 | 10.834 | 18.22 | 0.2761 | [0.2394, 0.3144] | 1.158 | 0.1714 | 0.2006 | 0.5469 |
| E2 | LoRA LLM + projector | 142,135,296 | 8.868 | 18.32 | 0.3171 | [0.2773, 0.3567] | 1.077 | 0.1410 | 0.2647 | 0.5843 |
| E3 | LoRA LLM + projector + LoRA ViT | 148,426,752 | 12.705 | 22.97 | 0.4100 | [0.3685, 0.4514] | 0.894 | 0.2922 | 0.3591 | 0.6194 |
| E4 | E3 + LoRA rank 64 (rank 4/64 được so trên validation) | 266,391,552 | 15.226 | 23.24 | 0.4322 | [0.3861, 0.4746] | 0.913 | 0.3084 | 0.3669 | 0.6788 |

E0–E4 dùng 311 ảnh test, cùng split và cùng phiên bản dữ liệu. E4 đã huấn luyện cả rank 4 và rank 64; rank 64 được chọn trên 40 ảnh validation (F1 micro 0.3763), sau đó chỉ rank được chọn mới được chấm test. E1–E3 trên cùng validation có F1 lần lượt 0.1979, 0.2723 và 0.4222, nên E3 là setup gốc cho ablation.

## 10 câu hỏi

### Câu 1 — Kiến trúc

Theo quy mô của model trong GUIDE, Qwen3-VL-4B có khoảng 4.0B tham số cho LLM và embedding, 415M cho phần lõi ViT, và 109M cho projector (`visual.merger` khoảng 27M, `visual.deepstack_merger_list` khoảng 82M), tổng cộng khoảng 4.44B. E1 cập nhật LoRA trên 252 module của LLM; E2 thêm cập nhật đầy đủ projector; E3 thêm LoRA cho 96 module ViT, tổng 348 module LoRA. Số trainable thực tế trong run là 33.03M, 142.14M và 148.43M tương ứng.

### Câu 2 — E0

`parse_valid_rate` của E0 là **0.9518** (296/311 output hợp lệ). Hai lỗi nổi bật là output JSON bị cắt/hỏng (15 output không parse được) và over-detection, đặc biệt nhiều box crack nhỏ, lặp trên cùng vùng. Ví dụ thật từ ảnh `vlcsnap-2025-02-18-23h12m24s456`:

```text
[{"label":"crack","bbox_2d":[489,450,520,470]},{"label":"crack","bbox_2d":[659,610,689,630]}, ... ,{"label":"crack","bbox_2d":[460,500,480,520]},{"label":"crack","bbox_2d":
```

Output dài 424 ký tự và kết thúc giữa một trường `bbox_2d`, nên parser báo `Expecting value` ở ký tự 425. Trên toàn bộ test, E0 có 364 FP và 535 FN; riêng crack có 303 FP và F1 chỉ 0.0327.

### Câu 3 — Thang tọa độ 0–1000

Chấm nhầm các tọa độ chuẩn hoá 0–1000 như pixel làm F1 giảm từ **0.0965 xuống 0.0000**; evaluator ghi nhận 412 FP và 583 FN. Đã kiểm chứng bằng cách giữ nguyên `e0_predictions.jsonl` và dataset, rồi chấm lại với `--coord pixel`. Tùy chọn đó chia tọa độ dự đoán cho chiều rộng/cao ảnh như thể chúng là pixel tuyệt đối, khiến box không còn trùng nhãn chuẩn hoá và không đạt IoU 0.5.

### Câu 4 — Mask loss

Chỉ tính loss trên phần assistant trả lời để tối ưu đúng chuỗi JSON box mà model cần tạo. Nếu tính cả prompt và token ảnh, phần mục tiêu sẽ chứa hướng dẫn người dùng và chuỗi token ảnh dài thay vì đáp án grounding; loss bị chi phối bởi nội dung không cần sinh, làm tín hiệu học vị trí box yếu đi. Bài kiểm tra processor xác nhận ảnh 640×360 tạo 220 image tokens và label được giải mã thành đáp án JSON cùng token kết thúc.

### Câu 5 — Projector, lượng tử hoá và float32

Các Linear 4-bit của bitsandbytes là trọng số lượng tử được đóng gói, không phải trọng số dense có thể mở `requires_grad` để fine-tune đầy đủ. Vì vậy cấu hình bỏ phần `visual` khỏi lượng tử hoá để projector giữ dạng dense; LLM vẫn dùng 4-bit và LoRA. Khi mở train projector ở E2/E3, các merger được chuyển sang float32 để cập nhật đầy đủ ổn định hơn với gradient nhỏ; khi nạp checkpoint để suy luận, chúng được ép về dtype của model.

### Câu 6 — Split theo phiên quay

Chia theo `session` giữ các frame gần kề của cùng video trong một split. Chia ngẫu nhiên có thể đặt các frame gần như trùng nhau vào train và test, làm F1 cao giả vì model đã thấy cùng cảnh. Split chuẩn có:

| Split | Ảnh | Pothole | Crack | Manhole |
|---|---:|---:|---:|---:|
| Train | 1,422 | 1,074 | 1,750 | 691 |
| Validation | 276 | 66 | 470 | 103 |
| Test | 311 | 121 | 299 | 163 |

### Câu 7 — So sánh E1/E2/E3

E1 đạt F1 **0.2761**, CI95 **[0.2394, 0.3144]**, VRAM **10.834 GB**, thời gian **18.22 phút**; E2 đạt **0.3171**, CI95 **[0.2773, 0.3567]**, **8.868 GB**, **18.32 phút**; E3 đạt **0.4100**, CI95 **[0.3685, 0.4514]**, **12.705 GB**, **22.97 phút**. E2 hơn E1 **0.0411 F1**, với khoảng 109.1M tham số huấn luyện bổ sung; thời gian gần như bằng nhau, còn VRAM quan sát được thấp hơn. E3 hơn E2 **0.0928 F1**, giảm FP/ảnh từ 1.077 xuống 0.894, nhưng dùng thêm khoảng 3.84 GB VRAM và 4.65 phút. E3 cho điểm cao nhất trong lần chạy này, đổi lại tốn VRAM hơn. CI95 E1 và E2 chồng lấp; CI E2 và E3 không chồng lấp, nhưng chưa tính bootstrap ghép cặp trên từng ảnh nên chưa khẳng định chênh lệch là có ý nghĩa thống kê.

### Câu 8 — Độ bất định với 311 ảnh test

Độ rộng CI95 F1 lần lượt khoảng **0.0750 (E1), 0.0795 (E2), 0.0828 (E3), 0.0885 (E4)**, tức sai số hai phía khoảng 0.038–0.044. Vì vậy các chênh lệch nhỏ, đặc biệt E1–E2 và E3–E4, chưa đủ để kết luận setup nào tốt hơn chắc chắn. CI E3/E4 chồng lấp; cần bootstrap ghép cặp hoặc tập test lớn hơn để kết luận chặt hơn.

### Câu 9 — Lỗi theo lớp

Ở E3, pothole là lớp yếu nhất: F1 **0.2922**, precision 0.3265, recall 0.2645, với 121 box thật; crack đạt 0.3591 trên 299 box và manhole đạt 0.6194 trên 163 box. E4 rank 64 tăng F1 các lớp lên 0.3084, 0.3669 và 0.6788; pothole vẫn yếu nhất với 86 FN và 71 FP. Pothole có ít mẫu hơn crack và vẫn có cả FP lẫn FN, nên có thể model gặp khó ở biến thiên hình dạng/kích thước và phân biệt với mảng vá hoặc bóng trên mặt đường. Đây là giả thuyết từ aggregate metrics; artifacts hiện tại chưa có phân tích phân bố kích thước box để xác nhận nguyên nhân.

Các ảnh dưới đây là cùng một ảnh test. Box xanh lá là ground truth; box đỏ/vàng/xanh dương là dự đoán pothole/crack/manhole. Ví dụ cho thấy E0 tạo box crack quá lớn, còn E3 định vị gọn hơn.

![E0 zero-shot trên ảnh test](report/figures/e0_vlcsnap-2025-02-18-23h14m08s861.png)

![E3 trên cùng ảnh test](report/figures/e3_vlcsnap-2025-02-18-23h14m08s861.png)

Ở ảnh test thứ hai, E0 bao phủ quá rộng vùng mặt đường; E3 thu hẹp dự đoán quanh vết nứt thật.

![E0 zero-shot trên ảnh test thứ hai](report/figures/e0_vlcsnap-2025-02-18-23h13m23s116.png)

![E3 trên ảnh test thứ hai](report/figures/e3_vlcsnap-2025-02-18-23h13m23s116.png)

Ở ảnh test thứ ba, E0 tiếp tục tạo box crack rộng; E3 định vị gần vết nứt hơn nhưng còn nhiều dự đoán dư.

![E0 zero-shot trên ảnh test thứ ba](report/figures/e0_vlcsnap-2025-02-18-23h16m51s863.png)

![E3 trên ảnh test thứ ba](report/figures/e3_vlcsnap-2025-02-18-23h16m51s863.png)

### Câu 10 — E4

E4 thay đúng một biến: LoRA `rank` (4 so với 64), đặt `alpha = 2 × rank`, trên setup gốc E3. Cả hai checkpoint hoàn thành 300 bước; rank 4 có 118,935,552 tham số huấn luyện, VRAM đỉnh 14.911 GB và 22.77 phút; rank 64 có 266,391,552 tham số, 15.226 GB và 23.24 phút. Trên cùng 40 ảnh validation, quy trình chọn rank 64 (F1 micro 0.3763), rồi đánh giá một lần trên test: F1 micro **0.4322** (CI95 **[0.3861, 0.4746]**), precision 0.4570, recall 0.4099, parse hợp lệ 100%, FP/ảnh 0.9132. So với E3, F1 điểm tăng 0.0222 và cả ba F1 theo lớp tăng nhẹ, nhưng FP/ảnh cũng tăng 0.0193; CI95 của E3 và E4 chồng lấp nên kết quả này chưa chứng minh rank 64 tốt hơn có ý nghĩa thống kê. Minh chứng đầy đủ nằm ở `report/evidence/e4.json`.

## Nhìn lại

E0 cho thấy sai số không chỉ đến từ JSON hỏng: output hợp lệ vẫn có nhiều box crack dư và miss nhiều box thật. Đánh giá bổ sung theo pixel scale cho thấy lỗi quy ước tọa độ đủ để làm F1 về 0 dù dự đoán và nhãn đều có box. Với E4, pothole vẫn là lớp khó nhất (86 FN); cần xem trực quan lỗi và kiểm tra box-size theo lớp trước khi quy nguyên nhân cho dữ liệu mất cân bằng. E4 tăng điểm so với E3 trong lần chạy này, nhưng số test và CI chồng lấp giới hạn kết luận.
