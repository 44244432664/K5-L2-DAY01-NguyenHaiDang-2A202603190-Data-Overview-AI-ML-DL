# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / Pytorch 2.11.0+cu130 / Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }


2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
	Record này nói về **một vật thể riêng**. Do:
	- Ngoài trường `task = "image_classification"` và mô hình `yolo11n-cls.pt` (model được dùng để phân loại ảnh), tôi còn dựa trên việc nhiều record có cùng `sample_id = "traffic"` nhưng mang các nhãn như `cab`, `minibus`, `police_van`,... được xếp theo `rank` và có các `score` giảm dần.

	- Cấu trúc này cho thấy mô hình đang đưa ra một danh sách các lớp dự đoán có mức độ phù hợp khác nhau đối với cùng một bức ảnh. Ví dụ, `cab` ở rank 1 là dự đoán được ưu tiên nhất, trong khi `minibus` và `police_van` là các lựa chọn có điểm thấp hơn.

	- Hơn nữa nếu đây là object detection, tôi kỳ vọng mỗi kết quả sẽ gắn với một đối tượng cụ thể thông qua thông tin vị trí, chẳng hạn bounding box. Trong dữ liệu hiện tại, không có thông tin như vậy.

Vì thế, tôi tin rằng đây là **image-level Top-K predictions**, chứ không phải danh sách các vật thể được phát hiện trong ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
	Danh sách tên lớp (class list) đi kèm với taxonomy

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
	Trong report phân loại ảnh (image classification), việc lưu cả `class_id`, `class_name` và `taxonomy_name` là để đảm bảo kết quả có thể được định danh chính xác, tái sử dụng và kiểm chứng (reproducibility), đặc biệt khi so sánh nhiều model hoặc nhiều phiên bản dataset.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
	Guideline cần nói rõ về cách ưu tiên chọn đối tương chính (vật thể hay chủ đề chính của ảnh,...), độ nổi bật để có thể xác định đối tương, diện tích chiếm của ảnh (hay metric khác) để ưu tiên gắn nhãn, nếu có ngữ cảnh hoặc caption đáng tin cậy thì có thể dùng để xác định nhãn hay không, khi có nhiều đối tượng ngang nhau thì ưu tiên chọn đối tượng nào, cách xử lý khi không có lớp nào trong taxonomy mô tả đúng ảnh

6. Vì sao không được lấy `score` của model làm ground truth?
	Vì `score` của model là kết quả của quá trình model được học dự đoán kết quả dựa trên các dữ đã học, do đó chỉ thể hiện mức độ tin tưởng của model vào dự đoán của chính nó, chứ không chứng minh dự đoán đó đúng, từ đó làm mất ý nghĩa của bước evaluation

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  }

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
	- góc trên-trái: 385.33, 69.24 (px)
	- góc dưới-phải: 498.92, 348.92 (px)
	- rộng: 113,58 px
	- cao: 279.68 px

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
threshold=0.35: 11 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl']
threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
	Người **reviewer** phải làm thêm việc tái kiểm tra nhãn nhằm tránh nhãn false-positive, nhận diện nhầm hay không, hộp nhãn có bị trùng lặp hay không, vị trí bounding box có bị lệch hay qía rộng/hẹp không
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
	**Quy tắc gán Bounding Box (XYXY):**
	Với mỗi vật thể nhìn thấy rõ và thuộc taxonomy quy định, vẽ **một bounding box hình chữ nhật nhỏ nhất có thể bao trọn phần vật thể nhìn thấy**, không cắt mất các phần có thể quan sát được và hạn chế chứa nền hoặc vật thể khác.
	- **Tọa độ:** Sử dụng định dạng `xyxy = [x_min, y_min, x_max, y_max]`, với gốc tọa độ ở góc trên bên trái ảnh.
	- **Vật thể bị che khuất:** Chỉ bao phần nhìn thấy, không tự suy đoán phần bị che.
	- **Vật thể bị cắt bởi mép ảnh:** Giới hạn box trong phạm vi ảnh.
	- **Nhiều vật thể:** Mỗi vật thể riêng biệt được gán một box, không gộp nhiều vật thể vào cùng một box.
	- **Vật thể không rõ ràng:** Đánh dấu `uncertain` để review thay vì đoán tọa độ hoặc nhãn.
	- **Box trùng lặp:** Không tạo nhiều box cho cùng một vật thể và cùng một nhãn.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
	Chiếc bát/đĩa ở sát mép trái, phía dưới ảnh, nơi một phần vật thể bị cắt khỏi khung hình.
	- Quyết định: Đưa vào guideline, không cần escalation lên Lab Coach.
	- Guideline bổ sung: `**Truncated Object Rule:** Khi một vật thể bị cắt bởi mép ảnh nhưng vẫn có thể xác định được lớp của nó, annotator chỉ vẽ bounding box bao quanh phần vật thể thực sự nhìn thấy, giới hạn tọa độ `xyxy` trong kích thước ảnh. Không suy đoán hoặc mở rộng box ra ngoài ảnh để bao phần bị mất.`
	- Lí do: chưa cần thiết, nên escalation khi có sự mơ hồ mà guideline hiện tại không giải quyết được.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0],...]

  "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-001",
    "class_id": 0,
    "class_name": "person",
    "score": 0.899318,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.45,
      66.44,
      498.02,
      348.58
    ],
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],...
    ]
  }

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
	So với box, polygon cho thêm thông tin về đường viền (contour) của ảnh, hình dạng và vùng diện tích thực tế của vật thể

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
	`instance_id` là mã định danh của từng cá thể vật thể riêng biệt trong một ảnh. Nó giúp phân biệt các vật thể với nhau, kể cả khi chúng thuộc cùng một class (ví dụ như class `person`), do đó, nó **không** phải class ID. Nó **không** phải tracking ID vì nó không nhất thiết phải giữ nguyên với cùng 1 người được xác định / label nhưng ở các frame khác nhau.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
	**Quy tắc vẽ biên Mask (Visible Boundary Rule):**
	Với mỗi vật thể, người gán nhãn phải vẽ biên mask bám sát **đường viền thực tế của phần vật thể nhìn thấy được**, không bao gồm nền hoặc các vật thể khác.
	- **Biên mask:** Bám sát hình dạng vật thể, kể cả các phần cong, góc cạnh hoặc bất quy tắc.
	- **Vật thể bị che khuất:** Chỉ đánh dấu phần nhìn thấy, không suy đoán phần bị che.
	- **Vật thể bị cắt mép ảnh:** Dừng biên mask tại mép ảnh, không mở rộng ra ngoài.
	- **Nhiều vật thể:** Mỗi cá thể có mask và `instance_id` riêng, kể cả khi cùng `class_id`.
	- **Vùng không rõ ràng:** Không tự đoán đường viền; đánh dấu `uncertain` để reviewer kiểm tra.

	**Mục tiêu:** Mask phải biểu diễn chính xác nhất có thể các pixel thuộc phần nhìn thấy của từng vật thể.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
	Phần tay người ở góc bên phải màn hình cùng với chén / dĩa khu vực đó.
	- Quyết định: Ưu tiên áp dụng guideline, chưa cần escalation
	- Lí do: Đây là tình huống có thể xử lý bằng quy tắc chung nên chưa cần escalation. Chỉ nên chuyển lên Lab Coach nếu biên giữa hai vật thể vẫn không xác định được sau khi kiểm tra kỹ, hoặc guideline không quy định cách xử lý vùng mơ hồ.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Mỗi ảnh có một nhãn chính (`class_id`, `class_name`) thuộc taxonomy `ImageNet-1K` | Ảnh `traffic` có thể chứa nhiều loại phương tiện. JSON dự đoán `cab` ở rank 1 với score 0.510915, ngoài ra còn có `minibus`, `police_van`,... nên cần xác định chủ thể chính thay vì mặc định Top-1 là đúng | Xác định chủ thể chính theo guideline, chọn một class phù hợp với taxonomy; đánh dấu ambiguous nếu không quyết định được | Kiểm tra nhãn có phù hợp nội dung ảnh và guideline không; kiểm tra sự nhất quán giữa các annotator; không lấy score của model làm ground truth |
| Phát hiện vật thể | Mỗi vật thể được gán một class và một bounding box dạng `xyxy` = [`x_min`, `y_min`, `x_max`, `y_max`] | Trong ảnh `kitchen`, một số bát/đĩa nằm sát mép trái và bị cắt một phần. JSON có 11 predictions ở threshold 0.35 nhưng chỉ còn 6 ở threshold 0.60, cho thấy số hộp cần kiểm tra thay đổi theo ngưỡng | Vẽ box bao sát phần vật thể nhìn thấy, gán đúng class, tách các vật thể riêng biệt và không suy đoán phần ngoài ảnh | Kiểm tra kĩ false positives, vật thể bị bỏ sót, class sai, box lệch/quá rộng và các box trùng lặp, đặc biệt với các prediction confidence thấp |
| Instance segmentation | Mỗi cá thể có `class_id`, `instance_id` và mask/polygon riêng thể hiện vùng pixel thuộc vật thể | Trong ảnh `kitchen`, vùng cánh tay người tiếp giáp với các dụng cụ bếp có thể khó phân biệt ranh giới. JSON có instance kitchen-001 thuộc class person, polygon gồm 348 điểm | Vẽ mask bám sát biên phần nhìn thấy, không bao gồm nền hay phần bị che; tách từng instance và đánh dấu uncertain nếu không rõ biên | Kiểm tra mask có tràn sang nền hoặc vật khác, bỏ sót vùng, nhầm instance hay không; escalation khi ranh giới không thể xác định theo guideline |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
	Chỉ sử dụng dữ liệu được cấp phép trong phạm vi bài lab, không công khai thông tin cá nhân hoặc dữ liệu nhạy cảm, và chỉ upload các file được yêu cầu lên repository theo hướng dẫn

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không thuộc phạm vi được phép, tôi sẽ dừng xử lý, không sử dụng hay chia sẻ dữ liệu đó, đồng thời báo cho Lab Coach hoặc giảng viên, hoặc người phụ trách để được hướng dẫn xử lý

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ x ] `REPORT.md` (file này đã điền)
- [ x ] `day1_lab_outputs/classification_predictions.json`
- [ x ] `day1_lab_outputs/detection_predictions.json`
- [ x ] `day1_lab_outputs/segmentation_predictions.json`
- [ x ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ x ] `day1_lab_outputs/visuals/classification_top5.png`
- [ x ]  `day1_lab_outputs/visuals/detection_predictions.png`
- [ x ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ x ] Ô validation cuối notebook báo `PASS`
- [ x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ x ] Đã nộp **link repository** trên LMS
