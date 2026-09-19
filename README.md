Tennis Highlight Summary

Hệ thống tự động tạo video highlight tennis sử dụng TrackNet, YOLOv8 và OpenCV để phát hiện bóng, người chơi, sân đấu và các tình huống rally trong trận đấu.

Giới thiệu đề tài

Dự án xây dựng một hệ thống xử lý video tennis nhằm tự động phát hiện các pha bóng nổi bật và tạo video highlight. Hệ thống kết hợp nhiều mô hình AI và thuật toán xử lý ảnh để nhận diện các thành phần trong trận đấu và xác định các sự kiện quan trọng.

Chức năng chính
Phát hiện và theo dõi bóng tennis bằng TrackNet.
Nhận diện sân tennis.
Nhận diện và theo dõi người chơi bằng YOLOv8.
Phát hiện điểm nảy của bóng (Bounce Detection).
Phát hiện Rally trong trận đấu.
Sinh MiniMap bằng phép biến đổi Homography.
Tự động cắt và xuất video highlight.
Cấu trúc dự án
TennisSummary/
├── app.py                     # Giao diện Web (Flask)
├── main.py                    # Pipeline xử lý video
├── TrackNet/                  # Module theo dõi bóng
├── PlayerDetector/            # Module nhận diện người chơi
├── TennisCourtDetector/       # Module nhận diện sân
├── minimap/                   # Homography và MiniMap
├── highlight/                 # Bounce, Rally, Video Cutter
├── static/
└── templates/
Quy trình huấn luyện TrackNet

Trong dự án, mô hình TrackNet được chuẩn bị và huấn luyện theo quy trình của repository gốc.

1. Chuẩn bị dữ liệu
Sử dụng bộ dữ liệu TrackNet chính thức gồm 10 trận đấu tennis.
Tổ chức dữ liệu theo đúng cấu trúc yêu cầu của TrackNet.
Bộ dữ liệu bao gồm ảnh các frame và file nhãn (Label.csv) cho từng clip.
2. Sinh Ground Truth

Tạo ảnh Ground Truth (Heatmap) từ dữ liệu gán nhãn bằng lệnh:

python gt_gen.py --path_input datasets/trackNet/images --path_output datasets/trackNet/gts

Kết quả tạo ra:

Thư mục gts/.
Hai file labels_train.csv và labels_val.csv.
3. Huấn luyện mô hình

Thực hiện huấn luyện mô hình bằng:

python main.py

Pipeline huấn luyện sử dụng:

labels_train.csv
labels_val.csv
Dữ liệu ảnh trong images/
Ground Truth trong gts/

Mô hình được lưu dưới dạng model_best.pt để sử dụng cho bước suy luận (Inference).

Lưu ý

Bộ dữ liệu gốc (khoảng 2.3 GB) không được đưa lên GitHub do dung lượng lớn. Repository chỉ bao gồm mã nguồn, cấu trúc dữ liệu và các file cần thiết để tái tạo quá trình huấn luyện.

Cách chạy dự án
Chạy giao diện Web
python app.py

Sau đó truy cập:

http://127.0.0.1:5000
Chạy pipeline tạo Highlight
python main.py
Công nghệ sử dụng
Python 3.8
PyTorch
OpenCV
Flask
YOLOv8
TrackNet
NumPy, Pandas
Kết quả đầu ra

Hệ thống nhận đầu vào là video trận đấu tennis và tạo ra:

Video đã theo dõi bóng.
MiniMap hiển thị vị trí bóng và người chơi.
Video Highlight các pha rally và bounce được cắt tự động.
