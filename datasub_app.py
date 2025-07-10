import cv2
import numpy as np
import streamlit as st

# Đọc ảnh
background = cv2.imread('background-2.jpg')
current = cv2.imread('current.jpg')

# Resize
background = cv2.resize(background, (640, 480))
current = cv2.resize(current, (640, 480))

# Chuyển sang xám
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

# Trừ ảnh
diff = cv2.absdiff(gray_bg, gray_current)

# Ngưỡng hóa
_, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Lọc nhiễu
kernel = np.ones((5, 5), np.uint8)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

# Trích foreground
foreground = cv2.bitwise_and(current, current, mask=fg_mask)

# Vector hóa (optional)
vec1 = gray_bg.flatten().astype(np.float32)
vec2 = gray_current.flatten().astype(np.float32)
cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 🎯 Hiển thị bằng Streamlit
st.title("Background Subtraction Demo")

st.subheader("Cosine Similarity giữa 2 ảnh:")
st.write(f"{cosine_sim:.4f}")

# Chuyển ảnh từ BGR -> RGB để hiển thị đúng màu
st.image(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), caption="Background", use_column_width=True)
st.image(cv2.cvtColor(current, cv2.COLOR_BGR2RGB), caption="Current Frame", use_column_width=True)
st.image(fg_mask, caption="Foreground Mask", use_column_width=True)
st.image(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB), caption="Extracted Foreground", use_column_width=True)
