import numpy as np
import cv2
import sys

filename = sys.argv[1]
cam = cv2.VideoCapture(filename)

# removes unnecessary parts of image and shrinks
def crop_image(image):
    image = image[180:400, :] # 220 by 640
    return cv2.resize(image, (44, 128), interpolation=cv2.INTER_AREA) # 44 by 128

# Take first frame and find corners in it
ret, old_frame = cam.read()
old_frame = crop_image(old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create feature array
X = []

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = crop_image(frame)
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magn, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    X.append(magn)

    # Now update the old frame to the current frame
    old_gray = new_gray

print(f'done getting frames from {filename}!')

X = np.stack(X)
name = filename.split('.')[0].split('/')[-1]
np.save(f'data/{name}_frames', X)

print('done writing frames!')

cv2.destroyAllWindows()
cam.release()