import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Tách nền & Ghép ảnh", layout="centered")
st.title("📸 Tách nền & Ghép vật thể vào nền")

# === Upload ảnh
st.subheader("📤 Tải ảnh lên")
bg_file = st.file_uploader("Ảnh nền (bg_img)", type=["jpg", "jpeg", "png"])
obj_file = st.file_uploader("Ảnh có vật thể (obj_img)", type=["jpg", "jpeg", "png"])

if bg_file and obj_file:
    try:
        # Đọc ảnh từ file buffer
        obj_bytes = np.frombuffer(obj_file.read(), np.uint8)
        bg_bytes = np.frombuffer(bg_file.read(), np.uint8)

        obj_img = cv2.imdecode(obj_bytes, cv2.IMREAD_COLOR)
        bg_img = cv2.imdecode(bg_bytes, cv2.IMREAD_COLOR)

        # Kiểm tra lỗi đọc ảnh
        if obj_img is None:
            st.error(
                "⚠️ Không thể đọc được **Ảnh có vật thể**. Vui lòng kiểm tra định dạng hoặc nội dung file."
            )
            st.stop()
        if bg_img is None:
            st.error(
                "⚠️ Không thể đọc được **Ảnh nền**. Vui lòng kiểm tra định dạng hoặc nội dung file."
            )
            st.stop()

        # Resize background để khớp với ảnh object
        obj_h, obj_w = obj_img.shape[:2]
        bg_img = cv2.resize(bg_img, (obj_w, obj_h))

        st.info("🖱 **Hướng dẫn:** Vẽ một đường bao quanh vật thể cần tách bằng chuột.")

        # Chuyển ảnh sang PIL để dùng cho canvas
        # Streamlit-drawable-canvas hoạt động tốt nhất với PIL Image
        obj_pil = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
        obj_pil_width, obj_pil_height = obj_pil.size
        # Canvas để vẽ vùng chọn
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",  # Màu fill của vùng vẽ
            stroke_width=5,  # Độ dày nét vẽ
            stroke_color="#FFFFFF",  # Màu nét vẽ (trắng để dễ thấy trên mọi nền)
            background_image=obj_pil,
            update_streamlit=True,  # Cập nhật Streamlit khi vẽ
            height=obj_pil_height,
            width=obj_pil_width,
            drawing_mode="freedraw",  # Chế độ vẽ tay tự do
            key="canvas",
        )

        # Nếu có dữ liệu ảnh vẽ từ canvas
        if canvas_result.image_data is not None:
            # Lấy kênh alpha từ dữ liệu canvas để tạo mask
            # Kênh alpha chứa thông tin về độ trong suốt/vùng đã vẽ
            alpha_channel = canvas_result.image_data[:, :, 3]

            # Tạo mask nhị phân từ kênh alpha. Ngưỡng 0 sẽ lấy tất cả các pixel có giá trị alpha > 0
            # Điều này giúp đảm bảo mọi phần vẽ đều được đưa vào mask
            mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[1].astype(
                np.uint8
            )

            # Tìm contours để fill vùng bên trong nét vẽ
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Tạo mask trắng hoàn toàn với vùng vẽ được fill kín
            filled_mask = np.zeros_like(mask)
            cv2.fillPoly(filled_mask, contours, 255)

            # Xử lý mask để làm sạch và làm đầy
            # Kernel cho các phép toán hình thái học
            kernel = np.ones((5, 5), np.uint8)

            # Morphological CLOSE: Đóng các lỗ nhỏ bên trong vùng vẽ và làm liền các vùng gần nhau
            filled_mask = cv2.morphologyEx(
                filled_mask, cv2.MORPH_CLOSE, kernel, iterations=2
            )
            # Dilate: Làm dày mask để đảm bảo vật thể được bao phủ hoàn toàn
            filled_mask = cv2.dilate(filled_mask, kernel, iterations=1)

            # Tách vật thể từ ảnh gốc bằng mask
            object_only = cv2.bitwise_and(obj_img, obj_img, mask=filled_mask)

            # Tạo phần nền mới: sử dụng ảnh nền và loại bỏ vùng vật thể
            mask_inv = cv2.bitwise_not(filled_mask)
            bg_part = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)

            # Ghép vật thể đã tách vào phần nền mới
            final = cv2.add(object_only, bg_part)

            # Hiển thị kết quả
            st.success("✅ Đã xử lý xong ảnh:")
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB), caption="Ảnh nền đã resize"
                )
                st.image(mask, caption="Mask tách vật thể")
            with col2:
                st.image(
                    cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB),
                    caption="Ảnh có vật thể gốc",
                )
                st.image(
                    cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="Kết quả ghép ảnh"
                )

        else:
            st.warning(
                "👉 Vui lòng vẽ vùng chứa vật thể trên ảnh **Ảnh có vật thể** để bắt đầu xử lý."
            )

    except Exception as e:
        st.error(f"⚠️ Đã xảy ra lỗi trong quá trình xử lý ảnh: `{e}`")
        st.exception(e)  # Hiển thị stack trace để debug chi tiết hơn

else:
    st.info("👈 Vui lòng tải lên cả hai ảnh (Ảnh nền và Ảnh có vật thể) để bắt đầu.")
