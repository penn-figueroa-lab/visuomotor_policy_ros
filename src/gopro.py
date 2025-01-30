import cv2
import numpy as np
from datetime import datetime

class GoProConfig:
    def __init__(self, device_name, frame_width, frame_height, fps, crop_rows=None, crop_cols=None):
        self.device_name = device_name
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.crop_rows = crop_rows or [-1, -1]
        self.crop_cols = crop_cols or [-1, -1]

class GoPro:
    def __init__(self):
        self.config = None
        self.cap = None
        self.image = None
        self.image_cropped = None
        self.time0 = None

    def initialize(self, time0, config: GoProConfig):
        """
        Initialize the GoPro pipeline.

        Args:
            time0: The start time.
            config: GoPro configuration.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        print("[GoPro] Initializing GoPro pipeline..")
        self.time0 = time0
        self.config = config

        # Open video capture
        self.cap = cv2.VideoCapture(config.device_name)
        if not self.cap.isOpened():
            print("\033[1;31mNo video stream detected. Check your device name config\033[0m")
            print(f"Current device name: {config.device_name}")
            return False

        # Configure the video capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

        # Test reading one frame
        print("Test reading a frame")
        ret, self.image = self.cap.read()
        if not ret or self.image is None:
            print("\033[1;31mTest reading failed\033[0m")
            print("  Possibility one: GoPro is not connected.")
            print("  Possibility two: Need to reset USB device.")
            print("    To do so, run 'lsusb | grep Elgato', which should give something like")
            print("      Bus 010 Device 005: ID 0fd9:008a Elgato Systems GmbH Elgato HD60 X")
            print("    Then run 'sudo ./USBRESET /dev/bus/usb/010/005'.")
            return False

        print("[GoPro] Pipeline started.")
        return True

    def next_rgb_frame_blocking(self):
        """
        Capture the next RGB frame from the GoPro camera.

        Returns:
            np.ndarray: The captured frame (cropped if configured), or an empty frame if no frame is detected.
        """
        ret, self.image = self.cap.read()
        if not ret or self.image is None:
            print("[GoPro] Empty frame. Terminate")
            return np.array([])

        if self.config.crop_rows[0] >= 0 and self.config.crop_cols[0] >= 0:
            self.image_cropped = self.image[
                self.config.crop_rows[0]:self.config.crop_rows[1],
                self.config.crop_cols[0]:self.config.crop_cols[1],
            ]
            return self.image_cropped
        else:
            return self.image

    def __del__(self):
        if self.cap:
            self.cap.release()
        print("[GoPro] Finishing..")

# Example usage
def main():
    # Define GoPro configuration
    config = GoProConfig(
        device_name="/dev/video0",  # Adjust this to your device
        frame_width=1920,
        frame_height=1080,
        fps=30,
        crop_rows=[100, 800],
        crop_cols=[200, 1200],
    )

    # Initialize GoPro
    gopro = GoPro()
    time0 = datetime.now()
    if not gopro.initialize(time0, config):
        print("Failed to initialize GoPro.")
        return

    # Capture frames in a loop
    try:
        while True:
            frame = gopro.next_rgb_frame_blocking()
            if frame.size == 0:
                break

            # Display the frame
            cv2.imshow("GoPro Frame", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
