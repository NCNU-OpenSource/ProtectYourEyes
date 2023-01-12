import cv2
import time
import numpy as np
import random
from playsound import playsound

def playAudio():
    music_list = ['credit', 'haaaa', 'chinaFighting', 'fighting', 'wtf', 'Bread', 'goatgirl', 'scream', 'uRTheBest', 'whatRUDoin', 'who']
    i = random.randrange(0, len(music_list))
    playsound('./Audio/'+music_list[i]+'.mp3')

def relaxUrEyes():
    goal = 30 # 需要按的目標數量
    text1 = "Please press enter "+str(goal)+ " times"
    img = np.zeros((2400, 3200, 3), np.uint8)
    cv2.putText(img, text1, (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
    cv2.putText(img, "You can do it", (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2)
    cv2.putText(img, "MOVE UP!", (450, 800), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 2)

    count = 0 # 數按鍵設了幾下
    while True:
        # 全螢幕
        cv2.namedWindow('Its Time To Break!', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Its Time To Break!', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 顯示圖片
        cv2.imshow('Its Time To Break!', img)
        key = cv2.waitKey(0)

        if key == 13:   # 如果按了 enter 鍵
            playAudio()
            count += 1

            # 顯示還剩餘幾下要按
            inputtext = str(goal-count) +" times left"
            org1,org2 = random.randrange(0,1500),random.randrange(50,900) # 隨機位置
            size = random.randrange(1,5) # 隨機文字大小
            r,g,b = random.randrange(0,255),random.randrange(0,255),random.randrange(0,255) # 隨機顏色
            cv2.putText(img, inputtext, (org1, org2), cv2.FONT_HERSHEY_SIMPLEX, size, (r, g, b), 2)
            if count == goal: # 如果按到目標次數則停止
                cv2.destroyAllWindows()
                break

def detectEyes():
    faceNum = 0 # 辨識出的臉部數量
    eyesNum = 0 # 辨識出的眼睛數量
    
    # 讀取要辨識的影像
    img = cv2.imread('./testPicture/testImg.jpg')

    # 將照片轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 讀入模型(臉和眼睛的)
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye_tree_eyeglasses.xml')

    # 將照片丟入模型辨識臉部
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    # 辨識到的臉的數量
    faceNum = len(faces)

    for (x,y,w,h) in faces:
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = img[y:y+h, x:x+w]
    
       # 在辨識為臉的區域辨識眼睛 (roi_)
       eyes = eye_cascade.detectMultiScale(roi_gray)
       eyesNum = len(eyes)

    if faceNum == 0 | eyesNum == 0:
        return 0
    else:
        return 1
    cv2.destroyAllWindows()


def snapShotCt(camera_idx, trans_time):
    # 目前累積的被辨識為盯著螢幕的狀態的次數
    accumulation_time = 0
    
    while accumulation_time < trans_time:
        cap = cv2.VideoCapture(camera_idx)
        ret, frame = cap.read()
        cv2.imwrite("./testPicture/testImg.jpg", frame)
        accumulation_time += detectEyes()
        print(accumulation_time)
        time.sleep(2)
        cap.release()
    relaxUrEyes()    

def main():
    # 預計用眼時間 (目前是設定以秒為單位，也可以設定為以分鐘)
    set_time = int(input()) 
    # 將用眼時間轉換成要觀測的次數
    trans_time = set_time/2
    snapShotCt(cv2.CAP_V4L2, trans_time)

if __name__ == "__main__":
    main()
