import cv2
import numpy as np
import streamlit as st

st.title("🧠 Background Subtraction + Ghép Vật Thể (Advanced)")

# Load ảnh
background = cv2.imread("background.jpg")
current = cv2.imread("current.jpg")

# Resize nếu cần
background = cv2.resize(background, (640, 480))
current = cv2.resize(current, (640, 480))

# Chuyển về ảnh xám
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

# Tạo ảnh mask bằng hiệu ảnh
diff = cv2.absdiff(gray_bg, gray_current)

# 🎯 Tùy chỉnh ngưỡng threshold
THRESH_VALUE = 50  # cao hơn để loại bỏ nhiễu nhỏ
_, mask = cv2.threshold(diff, THRESH_VALUE, 255, cv2.THRESH_BINARY)

# 🎯 Làm mượt mask bằng Gaussian blur
mask_blur = cv2_
