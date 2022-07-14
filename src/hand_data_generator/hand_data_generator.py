from collections import defaultdict

import cv2

from modules.hand_detector import HandDetector


class HandDataGenerator:
    """
    Generates Hand Data for Sign Language
    """

    def __init__(self):
        self.hand_data = defaultdict(lambda: defaultdict(int))
