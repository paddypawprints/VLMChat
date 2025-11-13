import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Union # Import Tuple, List, Union

# --- Named Colors (BGR format) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_PURPLE = (128, 0, 128)
COLOR_ORANGE = (0, 165, 255)
# --- End Named Colors ---

class ImageViewer:
    """
    A simple class to display images in a single, persistent window.
    
    This class uses OpenCV for rendering.
    """
    def __init__(self, window_name: str = "Image Viewer"):
        """
        Initializes the viewer and creates a named window.
        
        Args:
            window_name: The name to display on the window's title bar.
        """
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._is_window_visible = True
        self._current_image: Optional[np.ndarray] = None # Stores the last image shown

    def show(self, image: Optional[np.ndarray] = None, wait_ms: int = 1):
        """
        Displays a new image in the window, replacing the previous one.
        If no image is provided, the current image is redisplayed.

        Args:
            image: The image to display (as a NumPy array). If None, the 
                   last displayed image (if any) is redisplayed.
            wait_ms: Milliseconds to wait for a key press. This is 
                     necessary for the OpenCV GUI to refresh. 
                     Set to 0 to wait indefinitely.
        """
        if not self._is_window_visible:
            print("Window is closed. Cannot show image.")
            return

        if image is not None:
            self._current_image = image.copy() # Store a copy of the new image
        elif self._current_image is None:
            print("No image provided and no current image to display.")
            return

        try:
            cv2.imshow(self.window_name, self._current_image)
            
            key = cv2.waitKey(wait_ms)

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self._is_window_visible = False

        except cv2.error as e:
            print(f"OpenCV Error: {e}")
            print("This can happen if the window was closed unexpectedly.")
            self._is_window_visible = False

    def draw_box(self, 
                 image: np.ndarray, 
                 box: Tuple[int, int, int, int], 
                 color: tuple[int, int, int] = COLOR_GREEN, 
                 thickness: int = 2,
                 label: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        """
        Draws a rectangle on a copy of the image, with optional text labels.

        Args:
            image: The source image (NumPy array).
            box: The box coordinates as (x_min, y_min, x_max, y_max).
            color: The color of the rectangle in BGR format (e.g., (0, 255, 0) for green).
            thickness: The thickness of the rectangle line.
            label: Optional text label (or list of labels) to draw.

        Returns:
            A new image (NumPy array) with the rectangle and label drawn on it.
        """
        img_with_box = image.copy() # Always work on a copy

        # Extract points from the box tuple
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        
        cv2.rectangle(img_with_box, pt1, pt2, color, thickness)

        if label:
            # Ensure labels are in a list to handle both str and List[str]
            if isinstance(label, str):
                labels = [label]
            else:
                labels = label

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            # Start position for the first label
            text_x = pt1[0]
            text_y = pt1[1] - 10 # 10 pixels above the box
            line_height = 0 # To store height of a line

            for i, line in enumerate(labels):
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                line_height = text_size[1] + 5 # Height of text + 5px spacing

                # Calculate Y position for this line
                current_text_y = text_y + (i * line_height)

                # Ensure text is not drawn outside the image boundary
                if current_text_y < line_height: # If too close to top
                    # Move all text to be inside the box
                    text_y = pt1[1] + line_height # Reset base Y to be inside
                    current_text_y = text_y + (i * line_height)
                
                current_text_x = text_x
                if current_text_x < 0:
                    current_text_x = 0

                cv2.putText(img_with_box, line, (current_text_x, current_text_y), 
                            font, font_scale, color, font_thickness, cv2.LINE_AA)
            
        return img_with_box

    def is_visible(self) -> bool:
        """Checks if the display window is currently open."""
        if self._is_window_visible:
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self._is_window_visible = False
            except cv2.error:
                self._is_window_visible = False
                
        return self._is_window_visible

    def close(self):
        """Closes the image window."""
        if self._is_window_visible:
            cv2.destroyWindow(self.window_name)
            self._is_window_visible = False
            cv2.waitKey(1) # Process the destroy event


# Example usage:
if __name__ == "__main__":
    from typing import Optional, Tuple, List, Union # Import types

    viewer = ImageViewer()

    image1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image1, "Original Image 1", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    image2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.putText(image2, "Original Image 2", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # --- Example of draw_box with label list ---
    # Define box coordinates
    box1 = (50, 50, 300, 200) # (x_min, y_min, x_max, y_max)
    image1_boxed_labeled = viewer.draw_box(image1, box1, 
                                            color=COLOR_GREEN, 
                                            thickness=2, 
                                            label="Object A") # Single label

    # Now, draw another box with multiple labels
    box2 = (350, 250, 600, 400)
    multi_labels = ["Object B", "Confidence: 95%", "ID: 123"]
    image1_double_boxed = viewer.draw_box(image1_boxed_labeled, box2,
                                           color=COLOR_BLUE, 
                                           thickness=3, 
                                           label=multi_labels) # List of labels
    
    # --- Example of show() with no image argument ---
    # Show image2 first
    viewer.show(image2, wait_ms=1000)
    
    # Now draw a box on the current image (image2) and show it again
    print("\nDrawing box on current image and refreshing display...")
    box3 = (10, 10, 100, 100)
    current_img_with_box = viewer.draw_box(viewer._current_image, box3, color=COLOR_RED, label="Small Box")
    viewer.show(current_img_with_box, wait_ms=1000)
    
    # --- End Examples ---

    images_to_display = [image1_double_boxed, image2] # Use the double-boxed image

    print("Displaying images. Press 'q' in the window or close it to exit.")
    
    try:
        for img in images_to_display:
            if not viewer.is_visible():
                print("Window closed by user. Exiting loop.")
                break
                
            viewer.show(img, wait_ms=1500) # Show for 1.5 seconds
            
        if viewer.is_visible():
            print("Displaying last image for 3 seconds...")
            start_time = time.time()
            while time.time() - start_time < 3 and viewer.is_visible():
                key = cv2.waitKey(30)
                if key & 0xFF == ord('q'):
                    break
        
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt.")
    finally:
        print("Closing window.")
        viewer.close()