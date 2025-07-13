import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class ObjectSegmentationComposite:
    def __init__(self):
        pass
    
    def load_image(self, image_path):
        """Tải ảnh từ đường dẫn"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Không thể tải ảnh từ {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Lỗi tải ảnh: {e}")
            return None
    
    def create_mask_from_color(self, image, target_color, tolerance=30):
        """Tạo mask từ màu sắc cụ thể"""
        # Chuyển đổi sang HSV để dễ tách màu
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Định nghĩa khoảng màu
        lower = np.array([max(0, target_color[0] - tolerance), 50, 50])
        upper = np.array([min(179, target_color[0] + tolerance), 255, 255])
        
        # Tạo mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Làm mịn mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def improve_mask(self, mask):
        """Cải thiện chất lượng mask"""
        # Áp dụng Gaussian blur để làm mịn
        mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Áp dụng morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask_improved = cv2.morphologyEx(mask_blurred, cv2.MORPH_CLOSE, kernel)
        mask_improved = cv2.morphologyEx(mask_improved, cv2.MORPH_OPEN, kernel)
        
        # Làm mịn cạnh
        mask_improved = cv2.medianBlur(mask_improved, 5)
        
        return mask_improved
    
    def fix_inverted_mask(self, mask):
        """Sửa mask bị đảo ngược"""
        # Kiểm tra xem mask có bị đảo ngược không
        # Nếu vùng trắng (255) chiếm ít hơn vùng đen (0) thì có thể bị đảo
        white_pixels = np.sum(mask == 255)
        black_pixels = np.sum(mask == 0)
        
        if white_pixels > black_pixels:
            # Mask có vẻ bị đảo ngược, đảo lại
            return cv2.bitwise_not(mask)
        return mask
    
    def extract_object_with_mask(self, image, mask):
        """Tách vật thể sử dụng mask"""
        # Đảm bảo mask có cùng kích thước với ảnh
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Tạo mask 3 kênh
        if len(mask.shape) == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            mask_3ch = mask
        
        # Chuẩn hóa mask về giá trị 0-1
        mask_normalized = mask_3ch.astype(np.float32) / 255.0
        
        # Áp dụng mask lên ảnh
        object_extracted = image.astype(np.float32) * mask_normalized
        
        return object_extracted.astype(np.uint8), mask_normalized
    
    def composite_images(self, background, object_img, mask, position=(0, 0)):
        """Ghép vật thể vào ảnh nền"""
        bg_height, bg_width = background.shape[:2]
        obj_height, obj_width = object_img.shape[:2]
        
        # Tính toán vị trí ghép
        x, y = position
        x = max(0, min(x, bg_width - obj_width))
        y = max(0, min(y, bg_height - obj_height))
        
        # Tạo bản copy của background
        result = background.copy()
        
        # Vùng cần ghép trên background
        bg_region = result[y:y+obj_height, x:x+obj_width]
        
        # Đảm bảo mask có đúng kích thước
        if len(mask.shape) == 2:
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
        else:
            mask_3ch = mask
        
        # Ghép ảnh sử dụng alpha blending
        blended = bg_region * (1 - mask_3ch) + object_img * mask_3ch
        
        # Gán vùng đã ghép vào kết quả
        result[y:y+obj_height, x:x+obj_width] = blended.astype(np.uint8)
        
        return result
    
    def process_images(self, source_img, background_img, mask_img=None):
        """
        Xử lý ảnh với numpy arrays (dành cho Streamlit)
        
        Args:
            source_img: numpy array của ảnh nguồn
            background_img: numpy array của ảnh nền
            mask_img: numpy array của mask (tùy chọn)
        
        Returns:
            tuple: (result_image, mask_used, extracted_object)
        """
        try:
            # Đảm bảo ảnh có định dạng đúng
            if source_img is None or background_img is None:
                return None, None, None
            
            # Nếu có mask được cung cấp
            if mask_img is not None:
                # Chuyển mask sang grayscale nếu cần
                if len(mask_img.shape) == 3:
                    mask = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
                else:
                    mask = mask_img
                
                # Sửa mask đảo ngược nếu cần
                mask = self.fix_inverted_mask(mask)
            else:
                # Tạo mask tự động (tách nền xanh lá)
                mask = self.create_mask_from_color(source_img, [60, 255, 255], tolerance=40)
            
            # Cải thiện mask
            mask = self.improve_mask(mask)
            
            # Tách vật thể
            object_extracted, mask_normalized = self.extract_object_with_mask(source_img, mask)
            
            # Resize background về kích thước phù hợp
            target_height, target_width = 600, 800
            bg_resized = cv2.resize(background_img, (target_width, target_height))
            
            # Resize object nếu quá lớn
            obj_h, obj_w = object_extracted.shape[:2]
            if obj_h > target_height * 0.8 or obj_w > target_width * 0.8:
                scale = min(target_height * 0.8 / obj_h, target_width * 0.8 / obj_w)
                new_w = int(obj_w * scale)
                new_h = int(obj_h * scale)
                object_extracted = cv2.resize(object_extracted, (new_w, new_h))
                mask_normalized = cv2.resize(mask_normalized, (new_w, new_h))
            
            # Tính toán vị trí ghép (giữa ảnh)
            bg_h, bg_w = bg_resized.shape[:2]
            obj_h, obj_w = object_extracted.shape[:2]
            position = ((bg_w - obj_w) // 2, (bg_h - obj_h) // 2)
            
            # Ghép ảnh
            result = self.composite_images(bg_resized, object_extracted, mask_normalized, position)
            
            return result, mask, object_extracted
            
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý: {e}")
            return None, None, None

# Hàm tiện ích cho Streamlit
def process_with_streamlit(source_image, background_image, mask_image=None):
    """
    Hàm tiện ích để xử lý ảnh trong Streamlit
    
    Args:
        source_image: PIL Image hoặc numpy array
        background_image: PIL Image hoặc numpy array  
        mask_image: PIL Image hoặc numpy array (tùy chọn)
    
    Returns:
        tuple: (result_image, mask_used, extracted_object)
    """
    processor = ObjectSegmentationComposite()
    
    # Chuyển đổi PIL Image sang numpy array nếu cần
    if hasattr(source_image, 'convert'):
        source_img = np.array(source_image.convert('RGB'))
    else:
        source_img = source_image
    
    if hasattr(background_image, 'convert'):
        background_img = np.array(background_image.convert('RGB'))
    else:
        background_img = background_image
    
    mask_img = None
    if mask_image is not None:
        if hasattr(mask_image, 'convert'):
            mask_img = np.array(mask_image.convert('RGB'))
        else:
            mask_img = mask_image
    
    return processor.process_images(source_img, background_img, mask_img)

# Hàm demo cho việc test
def demo_with_file_paths(source_path, background_path, mask_path=None, output_path=None):
    """
    Demo function với đường dẫn file
    """
    processor = ObjectSegmentationComposite()
    
    # Kiểm tra file tồn tại
    if not os.path.exists(source_path):
        print(f"Không tìm thấy file: {source_path}")
        return None, None, None
    
    if not os.path.exists(background_path):
        print(f"Không tìm thấy file: {background_path}")
        return None, None, None
    
    # Tải ảnh
    source_img = processor.load_image(source_path)
    background_img = processor.load_image(background_path)
    mask_img = None
    
    if mask_path and os.path.exists(mask_path):
        mask_img = processor.load_image(mask_path)
    
    # Xử lý
    result, mask, extracted = processor.process_images(source_img, background_img, mask_img)
    
    # Lưu kết quả nếu có đường dẫn output
    if result is not None and output_path:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"Kết quả đã được lưu tại: {output_path}")
    
    return result, mask, extracted

# Test function (chỉ chạy khi được gọi trực tiếp)
def test_function():
    print("Test function - cập nhật đường dẫn file để test:")
    print("demo_with_file_paths('source.jpg', 'background.jpg', 'mask.jpg', 'output.jpg')")
