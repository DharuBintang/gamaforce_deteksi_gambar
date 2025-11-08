import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def load_image(image_path):
    """Load gambar dari file lokal"""
    try:
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' tidak ditemukan!")
            return None
        
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Tidak bisa membaca file '{image_path}'")
            return None
            
        print(f"✓ Gambar berhasil dimuat: {image_path}")
        print(f"  Ukuran: {img.shape[1]}x{img.shape[0]} pixels")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def detect_objects(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detected_objects = []
    
    # 1. DETEKSI LANDZONE (Persegi Panjang Biru)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_blue:
        area = cv2.contourArea(contour)
        if area > 5000:  
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
            detected_objects.append({
                'type': 'LANDZONE',
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'color': (0, 255, 0),  
                'color2': (0, 200, 0)  
            })
    
    lower_orange = np.array([5, 80, 80])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_orange:
        area = cv2.contourArea(contour)
        if area > 5000:  
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
            detected_objects.append({
                'type': 'DROPZONE',
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'color': (0, 255, 0),  
                'color2': (0, 200, 0)  
            })

            roi_hsv = hsv[y:y+h, x:x+w]
            if roi_hsv.size == 0:
                continue

            lower_white_roi = np.array([0, 0, 180])    
            upper_white_roi = np.array([180, 60, 255]) 

            mask_white_roi = cv2.inRange(roi_hsv, lower_white_roi, upper_white_roi)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            mask_white_roi = cv2.morphologyEx(mask_white_roi, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_white_roi = cv2.morphologyEx(mask_white_roi, cv2.MORPH_CLOSE, kernel, iterations=2)

            mask_white_roi = cv2.GaussianBlur(mask_white_roi, (5,5), 0)

            contours_w, _ = cv2.findContours(mask_white_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_w:
                area_c = cv2.contourArea(c)
                if area_c < 300:   
                    continue

                perim = cv2.arcLength(c, True)
                circularity = 4 * np.pi * area_c / (perim*perim) if perim > 0 else 0

                if circularity < 0.50:
                    continue

                (rx, ry), radius = cv2.minEnclosingCircle(c)   
                rx, ry = int(rx), int(ry)
                radius = int(radius)

                if radius <= 0:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        rx = int(M["m10"]/M["m00"])
                        ry = int(M["m01"]/M["m00"])
                    else:
                        continue

                global_cx = x + rx
                global_cy = y + ry
                global_r = radius
                bx, by, bw, bh = cv2.boundingRect(c)
                global_bbox = (x + bx, y + by, bw, bh)

                detected_objects.append({
                    'type': 'BUCKET',
                    'bbox': global_bbox,
                    'center': (global_cx, global_cy),
                    'radius': global_r,
                    'color': (255, 0, 255),  
                    'color2': (80, 0, 80),   
                    'from_roi': True  
                })

    # 3. DETEKSI BUCKET (Lingkaran - Merah)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=1)

    # Deteksi lingkaran biru gelap
    # lower_dark_blue = np.array([100, 50, 50])
    # upper_dark_blue = np.array([130, 255, 150])
    # mask_dark_blue = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
    # mask_dark_blue = cv2.morphologyEx(mask_dark_blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)

    mask_buckets = mask_red

    contours_buckets, _ = cv2.findContours(mask_buckets, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_buckets:
        area = cv2.contourArea(contour)
        if area > 1000:  
            
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity > 0.6:  
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w//2, y + h//2
                detected_objects.append({
                    'type': 'BUCKET',
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'color': (0, 225, 0),    
                    'color2': (0, 200, 200), 
                    'from_roi': False
                })
    
    
    return detected_objects

def draw_detections(image, objects):
    result = image.copy()
    
    for obj in objects:
        x, y, w, h = obj['bbox']
        cx, cy = obj['center']
        color = obj['color']
        color2 = obj.get('color2', color)  
        obj_type = obj['type']
        
        is_roi_bucket = obj.get('from_roi', False)
        
        if not is_roi_bucket:
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        if 'radius' in obj:
            r = int(obj['radius'])
            cv2.circle(result, (cx, cy), r, color, 2)
        
        should_draw_cross = (obj_type == 'LANDZONE') or (obj_type == 'BUCKET')
        
        if should_draw_cross:
            cross_size = 10
            cv2.line(result, (cx-cross_size, cy), (cx+cross_size, cy), color2, 2)
            cv2.line(result, (cx, cy-cross_size), (cx, cy+cross_size), color2, 2)
        
        if not is_roi_bucket:
            label = obj_type
            cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result

def main():
    print("\nMasukkan nama file gambar:")
    
    image_path = input("\nNama file gambar: ").strip().strip('"').strip("'")
    
    print(f"\nMemuat gambar dari: {image_path}")
    image = load_image(image_path)
    
    if image is None:
        print("\n❌ Gagal memuat gambar!")
        return
    
    detected_objects = detect_objects(image)
    
    for i, obj in enumerate(detected_objects, 1):
        extra = ''
        if 'radius' in obj:
            extra = f", radius={obj['radius']}"
    
    result = draw_detections(image, detected_objects)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(result_rgb)
    axes[1].set_title('Result')
    axes[1].axis('off')
    
    output_filename = 'hasil_deteksi.png'
    cv2.imwrite(output_filename, result)
    print(f"\n✓ Hasil disimpan sebagai '{output_filename}'")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()