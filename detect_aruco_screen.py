'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4
'''

import dis
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
import mss
import mss.tools

# Define the screen capture area (left, top, width, height)
screen_capture_area = {"left": 0, "top": 0, "width": 1920, "height": 1080}  # Adjust width and height to match your screen resolution


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--screen", required=True, help="Set to True if using sceen capture")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["screen"].lower() == "true":
	time.sleep(2.0)
else:
	if args["screen"] is None:
		print("[Error] Video file location is not provided")
		sys.exit(1)

if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

with mss.mss() as sct:
	while True:

		frame_col = np.array(sct.grab(screen_capture_area))
		h,w,_ = frame_col.shape

		width=1000
		height = int(width*(h/w))
		frame_col = cv2.resize(frame_col, (width, height), interpolation=cv2.INTER_CUBIC)
		frame = cv2.cvtColor(frame_col, cv2.COLOR_BGR2GRAY)

		
		corners, ids, rejected = detector.detectMarkers(frame)
		print(ids)
		print(corners)
		detected_markers = aruco_display(corners, ids, rejected, frame_col)
		for i in range(len(corners)):
			# Calculate the apparent size in pixels
			side_lengths = [np.linalg.norm(corners[i][0][0] - corners[i][0][1]),
                            np.linalg.norm(corners[i][0][1] - corners[i][0][2]),
                            np.linalg.norm(corners[i][0][2] - corners[i][0][3]),
                            np.linalg.norm(corners[i][0][3] - corners[i][0][0])]
			
			apparent_size = max(side_lengths)

			# Calculate the distance using the formula
			distance = (20 * 110.14) / apparent_size  # F = 3.04 mm, S = 20 cm
			focal_length_dash = (apparent_size * 110 ) / 20 # 104.5cm from my laptop screen

			cv2.putText(detected_markers, f"{distance:.2f} cm", (int(corners[i][0][:, 0].mean()), int(corners[i][0][:, 1].mean()) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
			
		
			print(f"Tag {ids[i][0]} - Distance: {distance:.2f} cm")
			print(f"Tag {ids[i][0]} - Focal: {focal_length_dash:.2f} cm")

		cv2.imshow("Image", detected_markers)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

cv2.destroyAllWindows()
#video.release()