import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Tách nền & Ghép ảnh", layout="centered")
st.title("📸 Tách nền & Ghép vật thể vào nền")

# === Upload ảnh
st.subheader("📤 Tải ảnh lên")
bg_file = st.file_uploader("Ảnh nền (background)", type=["jpg", "jpeg", "png"])
cur_file = st.file_uploader("Ảnh có vật thể (current)", type=["jpg", "jpeg", "png"])

# === Khi có cả 2 ảnh
if bg_file and cur_file:
    try:
        # Đọc ảnh từ file buffer
        bg_bytes = np.asarray(bytearray(bg_file.read()), dtype=np.uint8)
        cur_bytes = np.asarray(bytearray(cur_file.read()), dtype=np.uint8)

        background = cv2.imdecode(bg_bytes, cv2.IMREAD_COLOR)
        current = cv2.imdecode(cur_bytes, cv2.IMREAD_COLOR)

        # Kiểm tra lỗi đọc ảnh
        if background is None or current is None:
            raise ValueError("Không thể đọc được ảnh. Kiểm tra định dạng hoặc nội dung file.")

        # Resize ảnh current để khớp kích thước background
        h_bg, w_bg = background.shape[:2]
        current = cv2.resize(current, (w_bg, h_bg))

        # Chuyển sang grayscale
        gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

        # Ngưỡng điều chỉnh
        threshold_value = st.slider("🔧 Ngưỡng tách nền", 0, 100, 50)

        # Tạo mask
        diff = cv2.absdiff(gray_bg, gray_current)
        _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

        # === Làm mượt để loại nhiễu nhỏ
        mask_blur = cv2.GaussianBlur(mask, (9, 9), 0)

        # === Morphology nâng cao để:
        # 1. Loại nhiễu (OPEN), 2. Lấp lỗ (CLOSE), 3. Mở rộng vật thể (DILATE)
        kernel = np.ones((7, 7), np.uint8)
        mask_clean = cv2.morphologyEx(mask_blur, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)
        # Tạo mask 3 kênh
        mask_3ch = cv2.merge([mask_clean]*3)
        inv_mask = cv2.bitwise_not(mask_clean)
        inv_mask_3ch = cv2.merge([inv_mask]*3)

        # Lấy phần foreground (vật thể)
        foreground = cv2.bitwise_and(current, cv2.merge([mask_clean]*3))

        # Lấy phần background không có vật thể
        inv_mask = cv2.bitwise_not(mask_clean)
        background_part = cv2.bitwise_and(background, cv2.merge([inv_mask]*3))

        # Ghép: vật thể đặt lên background
        final = cv2.add(background_part, foreground)

        # Hiển thị kết quả
        st.success("✅ Đã xử lý xong ảnh:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), caption="Ảnh nền", use_container_width=True)
            st.image(mask_clean, caption="Mask tách nền", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(current, cv2.COLOR_BGR2RGB), caption="Ảnh có vật thể", use_container_width=True)
            st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="Kết quả ghép ảnh", use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Lỗi khi xử lý ảnh: `{e}`")

else:
    st.info("👈 Vui lòng tải lên cả hai ảnh.")
