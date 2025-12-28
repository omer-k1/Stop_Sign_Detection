import cv2
import numpy as np
import os

input_folder = "../images"
output_folder = "output_images"

def detect_stop_sign():
    files = os.listdir(input_folder)

    for filename in files:
        if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
            continue

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Hata: {filename}.")
            continue

        # hsv formatına
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #alt aralık
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])

        #üst aralık
        lower_red2 = np.array([150, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Sadece istenilen renk aralığı
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Gürültü azaltma
        kernel = np.ones((5, 5), np.uint8)
        #hatalatı temizleme
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #tek parça
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # birden çok alan vardı en büyüğü aldım.
            largest_contour = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(largest_contour)
            #çok küçük alanlar dahil edilmedi.
            if area > 2000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center_x = int(x + w / 2)
                center_y = int(y + h / 2)

                cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)

                print(f"Görsel:{filename} Merkez Konumu: x={center_x}, y={center_y}")

        output_path = os.path.join(output_folder, "detected_" + filename)
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    detect_stop_sign()