import cv2
import time
import numpy as np
import random

def timeToRelax():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.putText(img, "Please press enter 20 times", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "You can do it", (140, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "RIGHT?", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

    # 判斷
    count = 0
    while True:
        goal = 20
        # 全螢幕
        cv2.namedWindow('Is Time To Break!', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Is Time To Break!', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 顯示圖片
        cv2.imshow('Is Time To Break!', img)
        key = cv2.waitKey(0)
        # 按空白鍵
        if key == 13:
            count += 1
            inputtext = str(goal-count) +" times left"
            org1,org2 = random.randrange(0,300),random.randrange(20,500) # 位置
            size = random.randrange(1,3) # 文字大小
            r,g,b = random.randrange(0,255),random.randrange(0,255),random.randrange(0,255) # 顏色
            cv2.putText(img, inputtext, (org1, org2), cv2.FONT_HERSHEY_SIMPLEX, size, (r, g, b), 2)
            if count == goal:
                cv2.destroyAllWindows()
                break
            
        elif key == 32:
            cv2.destroyAllWindows()
            break


def detectEyes():
    faceNum = 0
    eyesNum = 0
    # read input image
    img = cv2.imread('./testPicture/testImg.jpg')
    # img = cv2.imread('./testPicture/testImg1.jpg')

    # convert to grayscale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # read the haarcascade to detect the faces in an image
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # read the haarcascade to detect the eyes in an image
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye_tree_eyeglasses.xml')

    # detects faces in the input image
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    # When no faces detected, face_classifier returns and empty tuple
    faceNum = len(faces)
    print('Number of detected faces:', len(faces))

    # loop over the detected faces
    for (x,y,w,h) in faces:
       # print(x,y,w,h)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = img[y:y+h, x:x+w]
    
       # detects eyes of within the detected face area (roi)
       eyes = eye_cascade.detectMultiScale(roi_gray)
       eyesNum = len(eyes)
       print('Number of detected eyes:', len(eyes))
    
       # draw a rectangle around eyes
       for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

    # display the image with detected eyes
    # cv2.imshow('Eyes Detection',img)
    # cv2.waitKey(0)
    if faceNum == 0 | eyesNum == 0: # 如果沒偵測到臉或眼睛就 false
        return 0
    else:
        return 1
    cv2.destroyAllWindows()


def snapShotCt(camera_idx, trans_time):
    cap = cv2.VideoCapture(camera_idx)
    ret, frame = cap.read()
    accumulation_time = 0
    while accumulation_time < trans_time:
        cv2.imwrite("./testPicture/testImg.jpg", frame)
        accumulation_time += detectEyes()
        time.sleep(5)
        ret, frame = cap.read()
    print("relax ur eyes!")
    cap.release()

def main():
    # 以分鐘為單位輸入多久之後警報器會響
    set_time = int(input()) 
    trans_time = set_time*60/5 # 預計有幾次被發現在盯著螢幕 每五秒拍一次照
    snapShotCt(cv2.CAP_V4L2, trans_time)

if __name__ == "__main__":
    main()

