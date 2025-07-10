import cv2
import numpy as np

# Đọc 2 ảnh: ảnh nền và ảnh có vật thể (giả sử same size)
background = cv2.imread('background.jpg')
current = cv2.imread('current.jpg')

# Resize về cùng kích thước nếu cần
background = cv2.resize(background, (640, 480))
current = cv2.resize(current, (640, 480))

# Chuyển sang ảnh xám
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

# Tính hiệu ảnh (sự thay đổi) -> foreground thô
diff = cv2.absdiff(gray_bg, gray_current)

# Ngưỡng hóa (threshold) để chỉ lấy vùng khác biệt rõ
_, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Áp dụng morphological operation để lọc nhiễu
kernel = np.ones((5, 5), np.uint8)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

# Tạo ảnh foreground
foreground = cv2.bitwise_and(current, current, mask=fg_mask)

# --- Tùy chọn: xử lý vector hóa ---
# Vector hóa hai ảnh để đo độ tương đồng (cosine, Euclidean, v.v.)
vec1 = gray_bg.flatten().astype(np.float32)
vec2 = gray_current.flatten().astype(np.float32)

# Tính cosine similarity giữa 2 vector
cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"Cosine Similarity giữa 2 ảnh: {cosine_sim:.4f}")

# --- Hiển thị kết quả ---
cv2.imshow("Original Background", background)
cv2.imshow("Current Frame", current)
cv2.imshow("Foreground Mask", fg_mask)
cv2.imshow("Extracted Foreground", foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()
