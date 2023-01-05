###### tags: `1111` `lsa`
# ProtectYourEyes LSA 期末專題實作

## 發展理念
- 起因
    - 有時候用電腦處理事情的時候都會不知不覺的用眼過度，所以決定寫一支可以督促(強迫)用眼過度的你休息一下，以此來 Protect your eyes!
- 功能
    - 眼部辨識
        - 每 n 秒幫使用者拍一張照，看看能不能偵測到你的眼睛
            > n 可自由設定，目前設 2 秒
    - 提醒機制
        - 每當偵測已到規定使用時間，就會跳出螢幕保護程式
    - 解鎖休息方式
        - 槌 n 下巨大Enter鍵關閉螢幕保護程式
            > n 可自由設定，目前設 20 下



## Implementation Resources
### 實作環境
- Ubuntu 虛擬機環境
### 使用技術
- Python 
- OpenCV
### 所使用的設備材料

- [巨大 enter 鍵](https://shopee.tw/product/70003480/4362094203?smtt=0.30911880-1672760034.9)
![](https://i.imgur.com/qNc4x1p.jpg =300x)


## 實作過程
- 如果是虛擬機，要確認虛擬機有讀到鏡頭
    ![](https://i.imgur.com/j13vRez.png)
    - 如果讀不到鏡頭可能是擴充包的問題
        ![](https://i.imgur.com/U97YQu1.png =400x)
        > 要確認 Oracle 版本，下載相對應的擴充包ㄛ

- 下載 opencv (需要版本 4 以上的)
```=
sudo apt install opencv-python == 4.2.0
```
- 察看 opencv 版本
```=
python3
import cv2 
print(cv2.__version__)
```

- 下載播放音效的模組
```=
pip3 install playsound
```

- import 會用到的套件，主要是使用 opencv
```python=
import cv2
import time
import numpy as np
import random
from playsound import playsound
```

- main function
```python=
def main():
    # 預計用眼時間 (目前是設定以秒為單位，也可以設定為以分鐘)
    set_time = int(input()) 
    # 將用眼時間轉換成要觀測的次數
    trans_time = set_time/2
    snapShotCt(cv2.CAP_V4L2, trans_time)

if __name__ == "__main__":
    main()
```
### 音效
```python=
def playAudio():
    music_list = ['chinaFighting', 'fighting', 'wtf', 'Bread', 'goatgirl', 'scream', 'uRTheBest', 'whatRUDoin', 'who']
    i = random.randrange(0, len(music_list))
    playsound('./Audio/'+music_list[i]+'.mp3')
```

### 影像辨識
1. 拍照，這邊設定每兩秒拍一次
```python=
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
```
2. 辨識方法: 臉是否正面對鏡頭 + 眼睛是否兩顆都偵測到
```python=
def detectEyes():
    faceNum = 0 # 辨識出的臉部數量
    eyesNum = 0 # 辨識出的眼睛數量
    
    # 讀取要辨識的影像
    img = cv2.imread('./testPicture/testImg.jpg')

    # 將照片轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 讀入模型
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye_tree_eyeglasses.xml')

    # 將照片丟入模型辨識臉部
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    # 辨識到的臉的數量
    faceNum = len(faces)

    for (x, y, w, h) in faces:
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = img[y:y+h, x:x+w]
    
       # 在辨識為臉的區域辨識眼睛 (roi_)
       eyes = eye_cascade.detectMultiScale(roi_gray)
       eyesNum = len(eyes)

    if faceNum == 0 | eyesNum != 2:
        return 0
    else:
        return 1
    cv2.destroyAllWindows()
```


### 螢幕保護:
1. 設定最一開始啟用時會出現的字，以及解鎖時所需要按的次數
```python=
goal = 30 #需要按的目標數量
text1 = "Please press enter "+str(goal)+ " times"
img = np.zeros((2400, 3200, 3), np.uint8)
cv2.putText(img, text1, (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
cv2.putText(img, "You can do it", (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2)
cv2.putText(img, "MOVE UP!", (450, 800), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 2)
```

2. 使用 while 迴圈偵測按鍵
``` python= 
count = 0 # 數按鍵設了幾下
while True:
    # 全螢幕
    cv2.namedWindow('Is Time To Break!', cv2WINDOW_NORMAL) # 讓窗口可調整大小並保持比例
    cv2.setWindowProperty('Is Time To Break!', cv2WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # 將視窗的屬性設定為全螢幕
    # 顯示圖片
    cv2.imshow('Is Time To Break!', img)
    key = cv2.waitKey(0)
   
    if key == 13:  # 如果按了 enter 鍵(13 為 enter 的 ASCII Code)
        count += 1
        
        # 顯示還剩餘幾下要按
        inputtext = str(goal-count) + " times left"
        org1,org2 = ranom.randrange(0,300),random.randrange(20,500) # 隨機位置
        size = random.randrange(1,3) # 隨機文字大小
        r,g,b = ranom.randrange(0,255),random.randrange(0,255),random.rndrange(0,255) # 隨機顏色
        cv2.putText(img, inputtext, (org1, org2), cv2FONT_HERSHEY_SIMPLEX, size, (r, g, b), 2)
        if count == goal: # 如果按到目標次數則停止
            cv2.destroyAllWindows()
            break
```


## 使用方法 
1. 安裝所有程式檔案
2. 啟動程式
```=
python3 takePicNDetec.py
```
## 工作分配


| 組員   |   工作分配   |
| :------: |:------------:|
| 李虹   | 影像辨識、寫 github、報告 |
| 簡翎恩 | 測試拍照功能、音效播放、報告  |
| 余姵穎 |  螢幕保護、寫 github、PPT、報告 |
| 李昀婕 | 螢幕保護、影片、報告 |
|  孫若綺 | 出席線上會議 |

## REF
- 影像辨識
    - [How to detect eyes in an image using OpenCV Python](https://www.tutorialspoint.com/how-to-detect-eyes-in-an-image-using-opencv-python)
    - [opencv haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- 螢幕保護
    - [效果文字](https://ithelp.ithome.com.tw/articles/10237817 )
    - [Python OpenCV 繪製文字 putText](https://shengyu7697.github.io/python-opencv-puttext/)
    - [偵測鍵盤行為](https://steam.oxxostudio.tw/category/python/ai/opencv-keyboard.html)
- 拍照
    - [Python實現連續拍照功能](https://blog.csdn.net/qq_41915690/article/details/88777184)
- 撥放音效
    - [Play sound in Python](https://www.geeksforgeeks.org/play-sound-in-python/)

## 未來展望
- 搭配眼動儀作更精確的「是否盯著螢幕」的判斷
- 增加更多樣化的休息及解鎖方式
- 結合樹莓派使用


## 遇到的困難
- 樹莓派上無法順利下載 opencv
- 要用的模組沒辦法順利下載
- 不知道要怎麼判斷是否盯著螢幕 (而且還要做得出來)
- 原本休息方式是強制回到桌面，但如果回到桌面就讀不到鍵盤訊號
- 用到 Ubuntu 壞掉



## 致謝!! :heart:
- 惠霖 @Huei-Lin-Lin: debug N 陪伴
- 柏瑋 @@PengLaiRenOu: debug
- 漢偉 @UncleHanWei: 題材發想
- 姜媽: 音效提供
