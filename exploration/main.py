import cv2 as cv
# from utils import rescaleFrame

# capture = cv.VideoCapture(0)python
capture = cv.VideoCapture("./data/VIRAT Ground Dataset/videos_original/VIRAT_S_000001.mp4")

while True:
    isTrue, frame = capture.read()
    # frame_resized = rescaleFrame(frame)
    pt1 = (100,100)
    pt2 = (500,600)
    cv.rectangle(frame, pt1, pt2, (0,255,0), -1)
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()