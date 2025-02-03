import cv2
import numpy as np
from datetime import datetime
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge  # Added for ROS Image conversion

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
        self.bridge = CvBridge()  # Initialize CvBridge

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

        # Image Publisher
        self.pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)

        # Test reading one frame
        print("Test reading a frame")
        ret, self.image = self.cap.read()
        if not ret or self.image is None:
            print("\033[1;31mTest reading failed\033[0m")
            return False

        print("[GoPro] Pipeline started.")
        return True
    
    def publish_frames(self):
        """Continuously capture frames and publish them as ROS Image messages."""
        rate = rospy.Rate(self.config.fps)  # Use ROS rate to control FPS
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Failed to capture frame!")
                break

            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            # Publish the image
            self.pub.publish(ros_image)

            # Show the frame (optional for debugging)
            cv2.imshow("Publishing Image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

# Example usage
def run():
    rospy.init_node("image_publisher", anonymous=True)  # Initialize ROS node

    # Define GoPro configuration
    config = GoProConfig(
        device_name="/dev/video0",  # Adjust this to your device
        frame_width=1920,
        frame_height=1080,
        fps=30,
    )

    # Initialize GoPro
    gopro = GoPro()
    time0 = datetime.now()
    if not gopro.initialize(time0, config):
        print("Failed to initialize GoPro.")
        return

    # Capture frames in a loop
    gopro.publish_frames()

if __name__ == "__main__":
    run()
