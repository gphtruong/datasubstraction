import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Background Subtraction", layout="centered")
st.title("🎥 Ghép vật thể từ ảnh vào nền bằng Background Subtraction")

# === Load ảnh (phải có trong thư mục)
background = cv2.imread("background.jpg")
current = cv2.imread("current.jpg")

if background is None or current is None:
    st.error("❌ Không tìm thấy file 'background.jpg' hoặc 'current.jpg' trong thư mục.")
    st.stop()

# === Resize ảnh
background = cv2.resize(background, (640, 480))
current = cv2.resize(current, (640, 480))

# === Chuyển sang grayscale
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

# === Thanh trượt điều chỉnh threshold
threshold_value = st.slider("Ngưỡng phát hiện khác biệt (threshold)", 0, 100, 50)

# === Subtraction và tạo mask
diff = cv2.absdiff(gray_bg, gray_current)
_, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

# === Làm mượt + morphology để lọc nhiễu
mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask_blur, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

# === Tạo mask màu
mask_3ch = cv2.merge([mask_clean] * 3)

# === Trích vật thể từ current.jpg
foreground = cv2.bitwise_and(current, mask_3ch)

# === Lấy phần nền còn lại từ background
inv_mask = cv2.bitwise_not(mask_clean)
inv_mask_3ch = cv2.merge([inv_mask] * 3)
background_part = cv2.bitwise_and(background, inv_mask_3ch)

# === Ghép ảnh
final = cv2.add(background_part, foreground)

# === Hiển thị ảnh
col1, col2 = st.columns(2)
with col1:
    st.image(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), caption="Ảnh nền (background)", use_column_width=True)
    st.image(mask_clean, caption="Foreground Mask", use_column_width=True)
with col2:
    st.image(cv2.cvtColor(current, cv2.COLOR_BGR2RGB), caption="Ảnh có vật thể (current)", use_column_width=True)
    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="Ảnh sau khi ghép", use_column_width=True)
