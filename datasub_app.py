import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Background Subtraction", layout="centered")
st.title("📸 Ghép vật thể vào nền - Background Subtraction (Streamlit Optimized)")

# === Upload ảnh
bg_file = st.file_uploader("📤 Upload ảnh nền (background)", type=["jpg", "png"])
curr_file = st.file_uploader("📤 Upload ảnh có vật thể (current)", type=["jpg", "png"])

# Khi đã có đủ ảnh
if bg_file and curr_file:
    # Đọc ảnh từ buffer
    file_bytes_bg = np.asarray(bytearray(bg_file.read()), dtype=np.uint8)
    file_bytes_curr = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)

    background = cv2.imdecode(file_bytes_bg, cv2.IMREAD_COLOR)
    current = cv2.imdecode(file_bytes_curr, cv2.IMREAD_COLOR)

    # Resiz
    current = cv2.resize(current, (640, 480))

    # Chuyển sang xám
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # Slider threshold
    threshold_value = st.slider("Ngưỡng phát hiện khác biệt", 0, 100, 50)

    # Tạo mask
    diff = cv2.absdiff(gray_bg, gray_current)
    _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Làm mượt + morphology
    mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_blur, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

    # Tạo ảnh ghép
    mask_3ch = cv2.merge([mask_clean] * 3)
    inv_mask = cv2.bitwise_not(mask_clean)
    inv_mask_3ch = cv2.merge([inv_mask] * 3)

    foreground = cv2.bitwise_and(current, mask_3ch)
    background_part = cv2.bitwise_and(background, inv_mask_3ch)
    final = cv2.add(background_part, foreground)

    # Hiển thị ảnh
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), caption="Ảnh nền", use_container_width=True)
        st.image(mask_clean, caption="Mask làm sạch", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(current, cv2.COLOR_BGR2RGB), caption="Ảnh có vật thể", use_container_width=True)
        st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="Ảnh đã ghép", use_container_width=True)

else:
    st.info("👈 Vui lòng upload cả hai ảnh để bắt đầu.")
