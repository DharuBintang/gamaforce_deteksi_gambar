import cv2
import numpy as np

def segment_apple(input_path, output_path):    
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Tidak dapat membaca gambar dari {input_path}")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([165, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_hsv = cv2.bitwise_or(mask_red1, mask_red2)
    
    l, a, b = cv2.split(lab)
    _, mask_lab = cv2.threshold(a, 135, 255, cv2.THRESH_BINARY)
    
    _, _, v = cv2.split(hsv)
    _, mask_bright = cv2.threshold(v, 40, 255, cv2.THRESH_BINARY)
    
    mask_color = cv2.bitwise_and(mask_hsv, mask_lab)
    mask_combined = cv2.bitwise_and(mask_color, mask_bright)
    
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
    
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros(mask_combined.shape, dtype=np.uint8)
    
    if contours:
        valid_contours = []
        img_area = img.shape[0] * img.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > img_area * 0.01 and area < img_area * 0.6:

                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.4:
                        valid_contours.append(contour)
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
            
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
            
            final_mask = cv2.erode(final_mask, kernel_small, iterations=1)
            
            final_mask = cv2.dilate(final_mask, kernel_medium, iterations=2)
        else:
            print("Warning: Tidak ada contour valid yang ditemukan")
            final_mask = mask_combined
    else:
        print("Warning: Tidak ada contour yang ditemukan")
        final_mask = mask_combined
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    
    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = np.where(labels == largest_component, 255, 0).astype(np.uint8)
    
    cv2.imwrite(output_path, final_mask)
    print(f"Binary mask berhasil disimpan ke {output_path}")
    
    cv2.imshow('Gambar Asli', img)
    cv2.imshow('Binary Mask', final_mask)
    
    overlay = img.copy()
    overlay[final_mask == 255] = [0, 255, 0]  
    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imshow('Overlay Result', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("=" * 50)
    print("Program Segmentasi Apel - Binary Mask (Improved)")
    print("=" * 50)
    
    input_file = input("\nMasukkan nama file gambar input : ")
    output_file = input("Masukkan nama file output : ")
    
    segment_apple(input_file, output_file)


if __name__ == "__main__":
    main()