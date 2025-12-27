"""
COCO category definitions for object detection.

This module provides the CocoCategory enum representing the 80 standard
COCO object categories used by YOLO and other detection models.
"""

from enum import Enum
from typing import Optional


class CocoCategory(Enum):
    """
    Enum representing the 80 COCO object categories.
    
    Each member has an id (COCO class ID) and label (string name).
    Used for type-safe category handling throughout the pipeline.
    """

    # --- Enum Members (id, "label") ---
    PERSON = (0, "person")
    BICYCLE = (1, "bicycle")
    CAR = (2, "car")
    MOTORCYCLE = (3, "motorcycle")
    AIRPLANE = (4, "airplane")
    BUS = (5, "bus")
    TRAIN = (6, "train")
    TRUCK = (7, "truck")
    BOAT = (8, "boat")
    TRAFFIC_LIGHT = (9, "traffic light")
    FIRE_HYDRANT = (10, "fire hydrant")
    STOP_SIGN = (11, "stop sign")
    PARKING_METER = (12, "parking meter")
    BENCH = (13, "bench")
    BIRD = (14, "bird")
    CAT = (15, "cat")
    DOG = (16, "dog")
    HORSE = (17, "horse")
    SHEEP = (18, "sheep")
    COW = (19, "cow")
    ELEPHANT = (20, "elephant")
    BEAR = (21, "bear")
    ZEBRA = (22, "zebra")
    GIRAFFE = (23, "giraffe")
    BACKPACK = (24, "backpack")
    UMBRELLA = (25, "umbrella")
    HANDBAG = (26, "handbag")
    TIE = (27, "tie")
    SUITCASE = (28, "suitcase")
    FRISBEE = (29, "frisbee")
    SKIS = (30, "skis")
    SNOWBOARD = (31, "snowboard")
    SPORTS_BALL = (32, "sports ball")
    KITE = (33, "kite")
    BASEBALL_BAT = (34, "baseball bat")
    BASEBALL_GLOVE = (35, "baseball glove")
    SKATEBOARD = (36, "skateboard")
    SURFBOARD = (37, "surfboard")
    TENNIS_RACKET = (38, "tennis racket")
    BOTTLE = (39, "bottle")
    WINE_GLASS = (40, "wine glass")
    CUP = (41, "cup")
    FORK = (42, "fork")
    KNIFE = (43, "knife")
    SPOON = (44, "spoon")
    BOWL = (45, "bowl")
    BANANA = (46, "banana")
    APPLE = (47, "apple")
    SANDWICH = (48, "sandwich")
    ORANGE = (49, "orange")
    BROCCOLI = (50, "broccoli")
    CARROT = (51, "carrot")
    HOT_DOG = (52, "hot dog")
    PIZZA = (53, "pizza")
    DONUT = (54, "donut")
    CAKE = (55, "cake")
    CHAIR = (56, "chair")
    COUCH = (57, "couch")
    POTTED_PLANT = (58, "potted plant")
    BED = (59, "bed")
    DINING_TABLE = (60, "dining table")
    TOILET = (61, "toilet")
    TV = (62, "tv")
    LAPTOP = (63, "laptop")
    MOUSE = (64, "mouse")
    REMOTE = (65, "remote")
    KEYBOARD = (66, "keyboard")
    CELL_PHONE = (67, "cell phone")
    MICROWAVE = (68, "microwave")
    OVEN = (69, "oven")
    TOASTER = (70, "toaster")
    SINK = (71, "sink")
    REFRIGERATOR = (72, "refrigerator")
    BOOK = (73, "book")
    CLOCK = (74, "clock")
    VASE = (75, "vase")
    SCISSORS = (76, "scissors")
    TEDDY_BEAR = (77, "teddy bear")
    HAIR_DRIER = (78, "hair drier")
    TOOTHBRUSH = (79, "toothbrush")

    def __init__(self, id: int, label: str):
        """Initialize with id and label properties."""
        self.id = id
        self.label = label

    @property
    def string_value(self) -> str:
        """Returns the string label of the category."""
        return self.label

    @classmethod
    def from_string(cls, value: str) -> Optional['CocoCategory']:
        """
        Gets an enum member from its string label.
        
        Args:
            value: String label (e.g., "person", "car")
            
        Returns:
            CocoCategory member or None if not found
        """
        for member in cls:
            if member.label == value:
                return member
        return None

    @classmethod
    def from_id(cls, id_value: int) -> Optional['CocoCategory']:
        """
        Gets an enum member from its COCO class ID.
        
        Args:
            id_value: COCO class ID (0-79)
            
        Returns:
            CocoCategory member or None if not found
        """
        for member in cls:
            if member.id == id_value:
                return member
        return None

    def __str__(self) -> str:
        """String representation (returns label)."""
        return self.label

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"CocoCategory.{self.name}({self.id}, '{self.label}')"
