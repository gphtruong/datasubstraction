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
            raise ValueError(
                "Không thể đọc được ảnh. Kiểm tra định dạng hoặc nội dung file."
            )

        # Resize ảnh current để khớp kích thước background
        h_bg, w_bg = background.shape[:2]
        current = cv2.resize(current, (w_bg, h_bg))

        # Chuyển sang grayscale
        gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

        # Ngưỡng điều chỉnh
        threshold_value = st.slider("🔧 Ngưỡng tách nền", 0, 100, 50)

        # Tạo mask vật thể bằng subtraction
        diff = cv2.absdiff(gray_bg, gray_current)
        _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

        # Xử lý mask để làm rõ vật thể
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Tạo mask màu 3 kênh và tách vật thể
        mask_3ch = cv2.merge([mask] * 3)
        foreground_only = cv2.bitwise_and(current, mask_3ch)

        # Lấy phần foreground (vật thể)
        foreground = cv2.bitwise_and(current, cv2.merge([mask] * 3))

        # Lấy phần background không có vật thể
        inv_mask = cv2.bitwise_not(mask)
        background_part = cv2.bitwise_and(background, cv2.merge([inv_mask] * 3))

        # Ghép: vật thể đặt lên background
        final = cv2.add(background_part, foreground)

        # Hiển thị kết quả
        st.success("✅ Đã xử lý xong ảnh:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                cv2.cvtColor(background, cv2.COLOR_BGR2RGB),
                caption="Ảnh nền",
            )
            st.image(
                mask,
                caption="Mask tách nền",
            )
        with col2:
            st.image(
                cv2.cvtColor(current, cv2.COLOR_BGR2RGB),
                caption="Ảnh có vật thể",
            )
            st.image(
                cv2.cvtColor(final, cv2.COLOR_BGR2RGB),
                caption="Kết quả ghép ảnh",
            )

    except Exception as e:
        st.error(f"⚠️ Lỗi khi xử lý ảnh: `{e}`")

else:
    st.info("👈 Vui lòng tải lên cả hai ảnh.")
