# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603867
- **Họ và tên:** Nguyễn Đình Minh
- **SEED** (in ở ô đầu notebook): 6379

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

# 2. Kết quả chính

Bảng tổng hợp từ lần chạy trên VS Code với **Ollama local**, sử dụng kết quả bạn đã gửi. Các LLM chỉ được đo trên tập **SUB 40 tin**; cột F1 test của chúng là F1 trên SUB, không phải toàn bộ tập test.

| Mô hình | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin (phí API ghi nhận) |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + Logistic Regression | 0.881 | 0.889 | 0.215 | 0.000 |
| 2. BoW + MLP | 0.957 | 0.947 | 0.546 | 0.000 |
| 3a. MiniLM + kNN | 0.920 | 0.974 | 21.811 | 0.000 |
| 3b. MiniLM + Logistic Regression | **0.968** | **0.974** | **9.505** | 0.000 |
| 3c. MiniLM + MLP head | 0.942 | 0.974 | [ĐIỀN p95 TỪ BẢNG ĐẦY ĐỦ] | 0.000 |
| 4a. Qwen2.5 0.5B zero-shot | 0.467 (SUB) | 0.467 | 251.46 | 0.000 |
| 4b. Llama3.2 3B zero-shot | 0.872 (SUB) | 0.872 | 572.33 | 0.000 |
| 4c. Qwen2.5 0.5B few-shot | 0.087 (SUB) | 0.087 | 189.45 | 0.000 |

**Nhận xét:** MiniLM + Logistic Regression có F1 cao nhất trên toàn bộ test (0.968) trong các mô hình được đánh giá đầy đủ, đạt F1 SUB 0.974 và p95 9.505 ms. BoW + MLP nhanh hơn nhiều nhưng F1 test thấp hơn (0.957). Kết quả LLM cho thấy model 3B tốt hơn model 0.5B ở zero-shot, nhưng cả hai kém MiniLM + Logistic Regression trên SUB 40 tin. Phí API bằng 0 là do chạy local, **không đồng nghĩa chi phí vận hành bằng 0**.

## 3. Dự đoán và thực tế

1. **Few-shot với LLM nhỏ:** Tôi dự đoán thêm bốn ví dụ sẽ giúp F1 tăng nhẹ so với zero-shot. Thực tế, Qwen2.5 0.5B giảm từ **0.467** (zero-shot) xuống **0.087** (few-shot), chênh **−0.380**. Có thể model rất nhỏ xử lý prompt dài, ví dụ mẫu và yêu cầu phân loại chưa tốt; cần xem từng nhãn dự đoán để xác định nguyên nhân chính xác. Few-shot không mặc nhiên cải thiện chất lượng.
2. **Chất lượng trên bộ 50 case khó:** Tôi dự đoán LLM zero-shot đạt khoảng **75–85% accuracy** và phiên bản prompt phòng thủ khoảng **80–90%**. Thực tế, LLM nhỏ zero-shot chỉ đạt **40% accuracy, F1 = 0.318**; prompt phòng thủ đạt **42% accuracy nhưng F1 = 0.000**. Bộ case cố tình chứa nội dung đánh lừa, spam viết tự nhiên và prompt injection; prompt phòng thủ còn khiến model bỏ sót nhiều spam, cho thấy accuracy không đủ để đánh giá.
3. **Learning rate 50:** Tôi dự đoán `lr=50` có thể dao động mạnh hoặc không hội tụ. Thực tế loss tăng vọt lúc đầu rồi giảm nhanh xuống **0.004 sau 300 epoch**, còn `lr=0.5` đạt **0.083** và `lr=0.001` đạt **0.628**. Learning rate lớn tạo bước cập nhật mạnh nhưng trong bài toán này vẫn hội tụ về training loss thấp; không nên lấy training loss làm bằng chứng tổng quát hóa tốt.

## 4. Phân tích 50 case khó

| Mô hình | F1 SUB 40 | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.889 | 0.378 | 54% |
| BoW + MLP | 0.947 | 0.500 | 60% |
| MiniLM + kNN | 0.974 | 0.714 | 68% |
| MiniLM + LogReg | 0.974 | 0.627 | 62% |
| MiniLM + MLP | 0.974 | 0.692 | 68% |
| LLM nhỏ zero-shot | 0.467 | 0.318 | 40% |
| LLM nhỏ + prompt phòng thủ | 0.467 (mốc zero-shot) | 0.000 | 42% |

**Mô hình tụt nhiều nhất trong so sánh F1 SUB với stress test:** BoW + LogReg giảm **0.511** (từ 0.889 xuống 0.378). Với BoW + LogReg, nhóm **spam giọng tự nhiên** chỉ đúng **20%**, nhóm **né từ khóa** chỉ đúng **17%**. MiniLM + kNN đạt F1 stress cao nhất (**0.714**) trong các phương án đã thử. LLM nhỏ zero-shot chỉ đúng **0%** nhóm spam giọng tự nhiên và prompt injection; prompt phòng thủ nâng nhóm injection lên 17% nhưng làm nhiều nhóm khác kém đi.

**Hai trường hợp cụ thể mà tất cả mô hình trong bảng stress test đều sai:**

- **Case #12 (spam giọng tự nhiên):** “hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for ...”. Tin được gắn nhãn **spam** nhưng mở đầu thân thiện như người quen, ít dấu hiệu spam nổi bật. BoW thiếu từ khóa, còn embedding/LLM cũng có thể bị đánh lừa bởi lối diễn đạt xã giao.
- **Case #13 (spam giọng tự nhiên):** “Hope you're well! Quick reminder that your photos from the event are ready, view & order prints ...”. Tin được gắn nhãn **spam**, nhưng giả như thông báo về ảnh sự kiện, khó phân biệt với trao đổi hoặc thông báo dịch vụ hợp lệ khi không có bối cảnh người gửi. Các mô hình có thể ưu tiên dấu hiệu ngôn ngữ bình thường nên dự đoán nhầm **ham**.

Các trường hợp này do người viết tay thiết kế để tìm điểm yếu, **không đại diện thống kê cho tỷ lệ spam ngoài thực tế**. Nên dùng để kiểm tra robustness và phân tích lỗi, không dùng riêng để ước lượng accuracy triển khai.

## 5. Thử thách cá nhân

**Đề #2 theo SEED 6379:** Giảm vocabulary còn 100 từ và tăng lên 5.000 từ bằng `build_vocab(..., max_size=...)`; huấn luyện lại Logistic Regression, báo F1 test và giải thích xu hướng.

**Cách làm:** Chỉ xây vocabulary từ `texts_tr` (không sử dụng dữ liệu test), tạo BoW cho train/test, huấn luyện Logistic Regression với cùng `lr=0.5`, `epochs=300`, sau đó tính Accuracy, Precision, Recall, F1 và thời gian train trên cùng tập test.

| Vocabulary | Accuracy | Precision | Recall | F1 test | Train (s) | Final training loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.964 | 0.946 | 0.745 | 0.833 | 0.061 | 0.122 |
| 2.000 | 0.974 | 1.000 | 0.787 | 0.881 | 1.364 | 0.083 |
| 5.000 | 0.974 | 1.000 | 0.787 | 0.881 | 3.641 | 0.081 |

**Giải thích:** Tăng từ 100 lên 2.000 từ giúp giữ nhiều đặc trưng hữu ích hơn, F1 tăng **0.048** và Recall tăng **0.042**. Tăng tiếp lên 5.000 từ không làm F1 test cải thiện (vẫn **0.881**), dù loss train giảm nhẹ 0.083 → 0.081 và thời gian train tăng khoảng **2.67 lần** so với 2.000 từ. Như vậy, thêm nhiều feature không nhất thiết thay đổi dự đoán đúng trên test; **2.000 từ là lựa chọn thực dụng hơn trong ba cấu hình đã thử**.

**Minh bạch sử dụng AI:** Tôi đã dùng AI hỗ trợ soạn code/gợi ý phân tích cho thử thách; tôi chạy và cung cấp kết quả để đối chiếu. Vì notebook ghi rõ mục này “không dùng AI”, cần trao đổi hoặc thực hiện lại độc lập theo quy định giảng viên trước khi nộp như một bài làm cá nhân.

## 6. Phương án cho sản phẩm thật

Yêu cầu **10 triệu SMS/ngày** tương ứng khoảng **116 tin/giây trung bình**, với **p95 dưới 50 ms/tin**. Tôi chọn **MiniLM + Logistic Regression** vì F1 test **0.968**, F1 SUB **0.974**, p95 **9.505 ms/tin**, thông lượng thử nghiệm khoảng **562 tin/giây**. Cách này cân bằng tốt chất lượng và độ trễ hơn LLM local trong kết quả hiện có. BoW + MLP là phương án nhẹ hơn, p95 **0.546 ms** và F1 test **0.957**, thích hợp khi ưu tiên tài nguyên hoặc dùng làm tầng lọc đầu.

Có thể triển khai **router**: BoW xử lý nhanh tin có xác suất rõ ràng; chuyển tin ở vùng không chắc chắn (`0.3 ≤ p ≤ 0.7`) sang MiniLM + LogReg. Tuy nhiên, cần hiệu chỉnh xác suất, đo tỷ lệ tin chuyển tầng và kiểm tra cả trường hợp BoW dự đoán sai nhưng rất tự tin. Việc dùng MiniLM chấp nhận độ trễ, bộ nhớ, công sức duy trì encoder cao hơn BoW để đổi lấy F1 tốt hơn. Các con số đo trong VS Code chỉ là benchmark của máy đang dùng: phải load-test riêng trên hạ tầng thực tế, nhất là giờ cao điểm, trước khi bảo đảm SLA.

Với **LLM**, các cấu hình local đã thử có p95 **189–572 ms**, vượt SLA. Phí API thống kê là **0 USD/1.000 tin** do dùng Ollama local; **không đủ dữ liệu để ước tính chi phí hạ tầng theo tháng**. Nếu dùng dịch vụ API trả phí, với khoảng **300 triệu tin/tháng**, công thức là `chi phí/tháng = 300.000 × giá USD/1.000 tin`, chưa bao gồm hạ tầng và chi phí khác; phải thay giá thật của nhà cung cấp trước khi tính tiền.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 4 — LLM | Hướng dẫn cấu hình Ollama và các hàm gọi/phân loại LLM. | Gợi ý `call_openai()`, prompt, parse nhãn và batch inference. | Chạy kiểm tra 4.1–4.4 và đối chiếu cấu hình Ollama local. |
| 4 — Few-shot | Giải thích chênh lệch zero-shot/few-shot. | Phân tích vì sao thêm ví dụ chưa chắc tăng F1. | Đối chiếu Qwen2.5 0.5B: F1 **0.467 → 0.087**. |
| 5 — Stress test | Phân tích lỗi trên 50 case khó và prompt injection. | Hỗ trợ tóm tắt kết quả, nhận diện nhóm yếu, cách phòng thủ. | Đối chiếu bảng F1, heatmap và case #12, #13. |
| 5 — Challenge | Hỗ trợ code so sánh vocabulary 100/2.000/5.000. | Gợi ý vòng lặp huấn luyện và phân tích F1. | Tự chạy thực nghiệm, thu được F1 **0.833 / 0.881 / 0.881**; khai báo AI đã hỗ trợ. |

**Điều đã học:** Tôi hiểu rõ hơn rằng F1 quan trọng hơn accuracy đơn lẻ khi lớp spam ít và việc bỏ sót spam có ý nghĩa lớn. Mô hình phức tạp hơn, vocab lớn hơn hay thêm few-shot chưa chắc cho kết quả tốt hơn; cần đo trên dữ liệu độc lập và các tình huống khó. Cuối cùng, chọn mô hình cho sản phẩm phải cân nhắc đồng thời chất lượng, độ trễ, thông lượng, chi phí và độ tin cậy của phép đo.
