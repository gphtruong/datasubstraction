import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    
    def auto_segment_and_composite(self, source_path, background_path, output_path=None):
        """Tự động tách và ghép ảnh"""
        # Tải ảnh
        source_img = self.load_image(source_path)
        background_img = self.load_image(background_path)
        
        if source_img is None or background_img is None:
            return None
        
        # Tạo mask tự động (ví dụ: tách nền xanh lá)
        # Bạn có thể điều chỉnh màu và tolerance theo nhu cầu
        mask = self.create_mask_from_color(source_img, [60, 255, 255], tolerance=40)
        
        # Cải thiện mask
        mask = self.improve_mask(mask)
        
        # Kiểm tra và sửa mask đảo ngược
        mask = self.fix_inverted_mask(mask)
        
        # Tách vật thể
        object_extracted, mask_normalized = self.extract_object_with_mask(source_img, mask)
        
        # Resize background nếu cần
        bg_resized = cv2.resize(background_img, (800, 600))
        
        # Tính toán vị trí ghép (giữa ảnh)
        bg_h, bg_w = bg_resized.shape[:2]
        obj_h, obj_w = object_extracted.shape[:2]
        position = ((bg_w - obj_w) // 2, (bg_h - obj_h) // 2)
        
        # Ghép ảnh
        result = self.composite_images(bg_resized, object_extracted, mask_normalized, position)
        
        # Lưu kết quả
        if output_path:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
        
        return result, mask, object_extracted
    
    def visualize_results(self, original, mask, extracted, composite):
        """Hiển thị kết quả"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Ảnh gốc')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Mask')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(extracted)
        axes[1, 0].set_title('Vật thể đã tách')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(composite)
        axes[1, 1].set_title('Kết quả ghép')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

# Sử dụng
def main():
    # Tạo instance
    processor = ObjectSegmentationComposite()
    
    # Đường dẫn ảnh (thay đổi theo ảnh của bạn)
    source_path = "path/to/your/source_image.jpg"
    background_path = "path/to/your/background_image.jpg"
    output_path = "path/to/output/composite_result.jpg"
    
    # Xử lý tự động
    result, mask, extracted = processor.auto_segment_and_composite(
        source_path, background_path, output_path
    )
    
    if result is not None:
        # Hiển thị kết quả
        source_img = processor.load_image(source_path)
        processor.visualize_results(source_img, mask, extracted, result)
        print(f"Kết quả đã được lưu tại: {output_path}")
    else:
        print("Có lỗi xảy ra trong quá trình xử lý")

# Hàm tiện ích để xử lý mask có sẵn
def process_with_existing_mask(source_path, background_path, mask_path, output_path):
    """Xử lý với mask có sẵn"""
    processor = ObjectSegmentationComposite()
    
    # Tải ảnh và mask
    source_img = processor.load_image(source_path)
    background_img = processor.load_image(background_path)
    mask_img = processor.load_image(mask_path)
    
    if source_img is None or background_img is None or mask_img is None:
        print("Không thể tải được một hoặc nhiều ảnh")
        return None
    
    # Chuyển mask sang grayscale
    mask = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    
    # Sửa mask đảo ngược nếu cần
    mask = processor.fix_inverted_mask(mask)
    
    # Cải thiện mask
    mask = processor.improve_mask(mask)
    
    # Tách vật thể
    object_extracted, mask_normalized = processor.extract_object_with_mask(source_img, mask)
    
    # Ghép vào nền
    bg_resized = cv2.resize(background_img, (800, 600))
    bg_h, bg_w = bg_resized.shape[:2]
    obj_h, obj_w = object_extracted.shape[:2]
    position = ((bg_w - obj_w) // 2, (bg_h - obj_h) // 2)
    
    result = processor.composite_images(bg_resized, object_extracted, mask_normalized, position)
    
    # Lưu kết quả
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    # Hiển thị kết quả
    processor.visualize_results(source_img, mask, object_extracted, result)
    
    return result

if __name__ == "__main__":
    # Chạy với mask tự động tạo
    main()
    
    # Hoặc chạy với mask có sẵn
    # process_with_existing_mask(
    #     "source.jpg", 
    #     "background.jpg", 
    #     "mask.jpg", 
    #     "result.jpg"
    # )
