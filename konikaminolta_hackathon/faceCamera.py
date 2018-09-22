import cv2
import requests
from time import sleep

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    speaking_counter = 0

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_default.xml"
    mouse_cascade_file = "haarcascade_mcs_mouth.xml"
    cascade = cv2.CascadeClassifier(cascade_file)
    mouse_cascade = cv2.CascadeClassifier(mouse_cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
        #文字の出力（会議名，ユーザ名）
        cv2.putText(img,'MeetingName', (5,40), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255, 255, 255))
        cv2.line(img,(2,55),(345,55),(255,255,255),2)
        cv2.putText(img,'UserName', (15,85), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255, 255, 255))
        # 検出した顔に印を付ける
        for (x, y, w, h) in face_list:
            color = (0, 0, 225)
            pen_w = 3
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
            #検出した口に印をつける
            face_lower = int(y*1.5)
            mouse_color = img[x:x+w, y:face_lower+h]
            mouse_gray = cv2.cvtColor(mouse_color, cv2.COLOR_BGR2GRAY)
            mouse = mouse_cascade.detectMultiScale(mouse_color, scaleFactor=1.1, minNeighbors=5, minSize=(100, 80))
            for (mx, my, mw, mh) in mouse:
                cv2.rectangle(mouse_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)
                # 口だけ切り出して二値化画像を保存
                threshold = 40 #二値化の閾値の設定
                for num, rect in enumerate(mouse):
                    x = rect[0]
                    y = rect[1]
                    width = rect[2]
                    height = rect[3]
                    dst = mouse_gray[y:y + height, x:x + width]
                    #二値化処理
                    ret,img_threshold = cv2.threshold(dst,threshold,255,cv2.THRESH_BINARY)
                    cv2.imwrite("./output/img_threshold.jpg",img_threshold)
                    binary_img = cv2.imread("./output/img_threshold.jpg")
                    cnt =0 #黒色領域の数を格納する変数
                    for val in binary_img.flat:
                        if val == 0:
                            cnt += 1
                    cv2.waitKey(1)
                    #二値化画像の黒色領域が何箇所あるかの判断→口が開いていれば黒いろ領域が多くなる＝発言している
                    if cnt > 600:
                        #print(speaking_counter, "Speaking!!")
                        speaking_counter += 1
                    #発言カウンターが10個貯まれば，サーバに秒数の送信
                    if speaking_counter == 10:
                        print("You spoke ", speaking_counter * 0.21, "second")
                        response = requests.post('https://jsondata.okiba.me/v1/json/m1NvW180922122602', data={'start': speaking_counter * 0.21})
                        #response = requests.post('http://requestbin.fullcontact.com/1au0lom1', data={'start': '2.1'})
                        speaking_counter = 0
                    else:
                        continue

        # フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
