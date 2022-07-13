import math
import cv2

import mediapipe as mp


class HandDetector:
    """
    Finds Hands using the mediapipe library.
    """

    def __init__(
        self,
        mode=False,
        max_hands=2,
        detection_confidence=0.5,
        min_track_confidence=0.5,
    ):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param max_hands: Maximum number of hands to detect
        :param detection_confidence: Minimum Detection Confidence Threshold
        :param min_track_confidence: Minimum Tracking Confidence Threshold
        """
        self.results = None
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.min_track_confidence = min_track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.min_track_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lm_list = []

    def find_hands(self, img, draw=True, flip_type=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :param flip_type: Defines if the hand image flip
        :return: Image with or without drawings
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_hands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(
                self.results.multi_handedness, self.results.multi_hand_landmarks
            ):
                my_hand = {}
                my_lm_list = []
                x_list = []
                y_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    my_lm_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                box_w, box_h = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, box_w, box_h
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                my_hand["lmList"] = my_lm_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)

                if flip_type:
                    if hand_type.classification[0].label == "Right":
                        my_hand["type"] = "Left"
                    else:
                        my_hand["type"] = "Right"
                else:
                    my_hand["type"] = hand_type.classification[0].label
                all_hands.append(my_hand)

                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
                    cv2.rectangle(
                        img,
                        (bbox[0] - 20, bbox[1] - 20),
                        (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                        (255, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        my_hand["type"],
                        (bbox[0] - 30, bbox[1] - 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 255),
                        2,
                    )
        if draw:
            return all_hands, img
        else:
            return all_hands
