import cv2
import os
import fcntl

# USB reset constant (_IO('U', 20))
USBDEVFS_RESET = ord('U') << (4 * 2) | 20


def reset_usb_device(device_path):
    """Reset the USB device."""
    try:
        with open(device_path, 'w') as usb_file:
            fcntl.ioctl(usb_file, USBDEVFS_RESET)
            print(f"Reset USB device {device_path} successfully.")
    except Exception as e:
        print(f"Failed to reset USB device {device_path}: {e}")


def main():
    # Path to your video device
    video_device = "/dev/video0"

    # Reset the USB device before initializing the video capture
    print("Resetting USB device...")
    reset_usb_device(video_device)

    # Initialize video capture
    print("Initializing video capture...")
    cap = cv2.VideoCapture(video_device)

    if not cap.isOpened():
        print("No video stream detected.")
        return

    # Create a window to display the video
    cv2.namedWindow("Video Player")

    try:
        while True:
            print("Reading new frame...")
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Empty frame. Terminate.")
                break

            # Show the video frame
            cv2.imshow("Video Player", frame)

            # Wait for 25ms or until 'Esc' key is pressed
            key = cv2.waitKey(25) & 0xFF
            if key == 27:  # ASCII value of 'Esc' key
                print("Escape key pressed. Exiting...")
                break
    finally:
        # Release the video capture and close all windows
        print("Video ended. Releasing resources.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
