import numpy as np
import cv2
img = cv2.imread('../../Gram Stain Images_ver2/C. krusei/IMG_20210623_165739__01.jpg')
img = img.astype(np.uint8)
cv2.imwrite("../../image.jpg", img);
#cv2.waitKey(0);
rot_img=np.rot90(img, 1) # Counter-Clockwise
rot_img = rot_img.astype(np.uint8)
#cv2.namedWindow("rotated image");
#cv2.imshow("rotated image", rot_img);
#cv2.waitKey(30);
cv2.imwrite('../../rotated.jpg',rot_img)