import os
import fcntl
import sys
import errno


# Constants for USB reset
USBDEVFS_RESET = 21780  # This corresponds to USBDEVFS_RESET in the Linux header

def reset_usb_device(device_path):
    """
    Reset a USB device by sending a USBDEVFS_RESET ioctl command.

    Args:
        device_path (str): Path to the USB device (e.g., /dev/bus/usb/001/002).

    Raises:
        OSError: If the operation fails.
    """
    try:
        # Open the USB device file
        with open(device_path, 'wb') as device:
            print(f"Resetting USB device {device_path}")
            # Send the USBDEVFS_RESET ioctl command
            fcntl.ioctl(device, USBDEVFS_RESET, 0)
            print("Reset successful")
    except OSError as e:
        if e.errno == errno.EPERM:
            print("Error: Permission denied. Run this script with sudo or as root.")
        else:
            print(f"Error resetting USB device: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reset_usb.py <device-path>")
        sys.exit(1)

    device_path = sys.argv[1]
    if not os.path.exists(device_path):
        print(f"Error: Device path {device_path} does not exist.")
        sys.exit(1)

    reset_usb_device(device_path)