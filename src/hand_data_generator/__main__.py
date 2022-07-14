import math
import time

import cv2
import numpy as np

from modules.hand_detector import HandDetector
from common.exceptions import VideoCanNotBeShown
from common.config import INPUT_DIR

hand_detector = HandDetector(max_hands=1)
cap = cv2.VideoCapture(0)
gap = 30
template_image_size = 300

# Change here for data collection
letter = "A"
letter_folder = f"{INPUT_DIR}/{letter}"
counter_for_letter_number = 0

while True:
    if cap.isOpened():
        success, frame = cap.read()
        hands, frame = hand_detector.find_hands(frame)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            template_image = (
                np.ones((template_image_size, template_image_size, 3), np.uint8) * 255
            )
            hand_image = frame[y - gap : y + h + gap, x - gap : x + w + gap]

            hand_image_shape = hand_image.shape
            hand_image_ratio = h / w

            if hand_image_ratio > 1:
                constant = template_image_size / h
                calculated_hand_image_width = math.ceil(constant * w)
                resized_hand_image = cv2.resize(
                    hand_image, (calculated_hand_image_width, template_image_size)
                )
                resized_hand_image_shape = resized_hand_image.shape
                width_gap = math.ceil(
                    (template_image_size - calculated_hand_image_width) / 2
                )
                template_image[
                    :, width_gap : calculated_hand_image_width + width_gap
                ] = resized_hand_image
            else:
                constant = template_image_size / w
                calculated_hand_image_height = math.ceil(constant * h)
                resized_hand_image = cv2.resize(
                    hand_image, (template_image_size, calculated_hand_image_height)
                )
                resized_hand_image_shape = resized_hand_image.shape
                height_gap = math.ceil(
                    (template_image_size - calculated_hand_image_height) / 2
                )
                template_image[
                    height_gap : calculated_hand_image_height + height_gap, :
                ] = resized_hand_image

            try:
                cv2.imshow("template_image", template_image)
            except VideoCanNotBeShown:
                pass

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter_for_letter_number += 1
            cv2.imwrite(f"{letter_folder}/{time.time()}.jpg", template_image)
            print(counter_for_letter_number)
