import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def ocr_core(img):
    # Preprocess the image

    # Perform OCR on the preprocessed image with custom configurations
    custom_config = r'--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
    text = pytesseract.image_to_string(img, config=custom_config)

    return text

def preprocess_image(img):

    #resizes the image to make the image more readable if small
    re = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


    # Convert to grayscale
    gray = cv2.cvtColor(re, cv2.COLOR_BGR2GRAY)



    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilation and Erosion
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded



path = "Images/dox.png"

img = cv2.imread(path)

img = preprocess_image(img)


print(ocr_core(img))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()