# Báo cáo lab: chọn tracker cho 5 video

**Thành viên:** Dương Văn Thành (02368), Hồ Ngọc Mai (02509)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

## 1. Cấu hình đã chọn

Mỗi video: tracker bạn nộp, `conf`, `iou`, điều bạn **nhìn thấy** trên video, và một cấu hình đã thử rồi loại.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.20 | 0.50 | Ban ngày ánh sáng tốt, Re-ID hỗ trợ giữ danh tính chuẩn khi người đi bộ che khuất ngắn hoặc đi chéo nhau; ít ID switch hơn, đạt HOTA cao nhất (29.16). | bytetrack (conf=0.30, iou=0.50) — HOTA chỉ đạt 26.91, bỏ sót một số người ở khoảng cách xa do conf hơi cao. |
| video_2 (phố đêm, tĩnh, rất đông) | bytetrack | 0.25 | 0.50 | Cảnh đêm mật độ người rất dày đặc, góc nhìn cao xuống chỉ thấy đỉnh đầu/vai. ByteTrack dùng liên kết 2 tầng phát hiện (high/low score) giúp bám tốt người bị che khuất mà không phụ thuộc vào Re-ID bị nhiễu do bóng tối. | botsort (conf=0.25, iou=0.50) — Re-ID trong điều kiện thiếu sáng và bóng đèn đường dễ sinh vector ngoại hình tương đồng, dẫn đến gán nhầm ID khi hai người đi sát nhau. |
| video_3 (camera di động, ảnh nhỏ) | ocsort | 0.30 | 0.50 | Camera chuyển động và FPS thấp làm giả định vận tốc tuyến tính của Kalman Filter thông thường bị trôi. OC-SORT dùng momentum và quan sát định hướng giúp duy trì track ổn định khi người chuyển hướng đột ngột. | bytetrack (conf=0.30, iou=0.50) — Bị đứt track và tạo ID mới liên tục khi camera xoay nhanh do mất dấu hộp bounding box ở frame kế tiếp. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.35 | 0.50 | Ánh sáng trong nhà rõ nét, người thay đổi kích thước khi camera tiến tới. BoT-SORT kết hợp bù chuyển động camera (GMC) cùng Re-ID giúp loại trừ hiện tượng bắt nhầm bóng phản chiếu trên kính và giữ ID tốt. | bytetrack (conf=0.35, iou=0.50) — Thiếu bù chuyển động nền và Re-ID nên dễ bị nhảy ID khi camera di chuyển tới gần đối tượng. |
| video_5 (trên xe bus, giao lộ đông) | ocsort | 0.30 | 0.50 | Xe rung giật mạnh ở giao lộ khiến vị trí người nhảy xô lệch qua từng frame. Thuật toán OC-SORT với cơ chế Observation-Centric Recovery (OCR) phục hồi vết dựa trên quan sát thực tế và kiểm tra góc hướng, giảm rung lắc track. | botsort (conf=0.30, iou=0.50) — Chi phí trích xuất Re-ID cao hơn, xe rung mạnh làm feature ngoại hình mờ nhòe gây giảm độ chính xác khi tái kết nối track. |

## 2. Số liệu video_1

Dán bảng HOTA / MOTA / IDF1 do `scripts/evaluate_practice.py` in ra.

```
HOTA: nop_bai_video1-pedestrian    HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
video_1                            29.16     18.823    45.49     19.427    77.562    48.231    83.015    83.247    29.664    35.62     77.805    27.714    
COMBINED                           29.16     18.823    45.49     19.427    77.562    48.231    83.015    83.247    29.664    35.62     77.805    27.714    

CLEAR: nop_bai_video1-pedestrian   MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag      
video_1                            20.596    80.894    20.731    22.889    91.384    12.903    20.968    66.129    16.223    4253      14328     401       25        8         13        41        75        
COMBINED                           20.596    80.894    20.731    22.889    91.384    12.903    20.968    66.129    16.223    4253      14328     401       25        8         13        41        75        

Identity: nop_bai_video1-pedestrianIDF1      IDR       IDP       IDTP      IDFN      IDFP      
video_1                            29.111    18.201    72.669    3382      15199     1272      
COMBINED                           29.111    18.201    72.669    3382      15199     1272      

Count: nop_bai_video1-pedestrian   Dets      GT_Dets   IDs       GT_IDs    
video_1                            4654      18581     54        62        
COMBINED                           4654      18581     54        62        
```

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

### Video 2 (Phố đêm, camera tĩnh trên cao, mật độ rất đông):
Trong video 2, việc lựa chọn `bytetrack` với ngưỡng `--conf 0.25` và `--iou 0.5` đem lại quỹ đạo ổn định và ít nhảy ID hơn rõ rệt so với các tracker có Re-ID. Nguyên nhân xuất phát từ đặc thù môi trường: ban đêm độ sáng thấp, người đi bộ chủ yếu mặc quần áo tối màu và góc nhìn từ trên cao khiến các đặc trưng ngoại hình từ mô hình Re-ID (OSNet) bị suy biến, dễ bị lẫn lộn giữa những người đi gần nhau. ByteTrack không phụ thuộc vào ngoại hình mà tập trung tận dụng các detection có điểm tin cậy thấp (low confidence detections) ở bước ghép đôi thứ hai, qua đó bám vết liên tục được những người bị che khuất một phần trong đám đông mà không tạo thêm ID rác.

### Video 3 & Video 5 (Camera di động, frame chậm & xe bus rung lắc):
Đối với video 3 và video 5, sự dịch chuyển của camera và độ rung lắc mạnh làm phá vỡ giả định vận tốc tuyến tính không đổi của bộ lọc Kalman tiêu chuẩn. Thuật toán `ocsort` thể hiện sự vượt trội khi giữ được ID xuyên suốt thời điểm camera chao đảo hoặc rung lắc đột ngột. Cơ chế Observation-Centric Online Smoothing (OCOS) cùng Observation-Centric Recovery (OCR) trong OC-SORT cập nhật lại trạng thái chuyển động dựa trên các quan sát thực tế và tính nhất quán về phương hướng chuyển động, thay vì tích lũy sai số dự đoán như trong SORT cổ điển hay ByteTrack. Nhờ vậy, hiện tượng gán nhầm hoặc phân mảnh track (fragmentation) khi camera giật lắc được giảm thiểu đáng kể.

## 4. Nếu có thêm thời gian

Nếu có thêm thời gian, nhóm sẽ thử nghiệm tinh chỉnh sâu hơn ngưỡng hai tầng của ByteTrack (`track_high_thresh` và `track_low_thresh`) cho riêng cảnh đêm đông đúc của video 2, đồng thời tích hợp mô hình Re-ID có khả năng thích ứng với điều kiện ánh sáng yếu (domain adaptation) hoặc áp dụng mô hình cân bằng chuyển động camera chuyên biệt (ECC camera motion compensation) cho video 5 để xử lý triệt để rung lắc do xe bus.
