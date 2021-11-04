import cv2
import imutils

cap = cv2.VideoCapture('D:\MaDoc\sample.avi')
rotate = 0
while True:
    ret, frame = cap.read()
    if ret:
        frame = imutils.rotate(frame, rotate)
        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 21)
        canny = cv2.Canny(frame, threshold1=20, threshold2=50)
        cv2.imshow('Frames', frame)
        cv2.imshow('Gray', gray)
        cv2.imshow('Thresh', thresh)
        cv2.imshow('Canny', canny)

    k = cv2.waitKey(1)
    if k == ord('a'):
        rotate = 90
    if k == ord('s'):
        rotate = 180
    if k == ord('d'):
        rotate = -90
    if k == ord('e'):
        rotate = 0
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
