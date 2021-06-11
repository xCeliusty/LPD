import cv2
from matplotlib import pyplot as plt
import easyocr
import imutils
import numpy as np
import datetime

image = cv2.imread("images/Cars53.png")
Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# To Display GrayScale Image
# plt.imshow(cv2.cvtColor(Gray, cv2.COLOR_BGR2RGB))

BiFilter = cv2.bilateralFilter(Gray, 9, 75, 75)
Edged = cv2.Canny(BiFilter, 50, 200)
# To Display Edged Image
# plt.imshow(cv2.cvtColor(Edged, cv2.COLOR_BGR2RGB))

Keys = cv2.findContours(Edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(Keys)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

location = None
for i in contours:
    temp = cv2.approxPolyDP(i, 10, True)
    if len(temp) == 4:
        location = temp
        break

location

mask = np.zeros(Gray.shape, np.uint8)
CroppedImage = cv2.drawContours(mask, [location], 0, 255, -1)
CroppedImage = cv2.bitwise_and(image, image, mask=mask)
# To Display Cropped Image
# plt.imshow(cv2.cvtColor(CroppedImage, cv2.COLOR_BGR2RGB))

reader = easyocr.Reader(['en'])
output = reader.readtext(CroppedImage)
output

text = output[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
result = cv2.putText(image, text=text, org=(temp[0][0][0], temp[1][0][1]+60), fontFace=font, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
result = cv2.rectangle(image, tuple(temp[0][0]), tuple(temp[2][0]), (0, 0, 255), 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
dt_string = datetime.datetime.now()
status = cv2.imwrite('Results/test.png', result)
plt.show()