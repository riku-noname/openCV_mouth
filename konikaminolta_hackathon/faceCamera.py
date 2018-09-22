import cv2
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
                threshold = 40
                for num, rect in enumerate(mouse):
                    x = rect[0]
                    y = rect[1]
                    width = rect[2]
                    height = rect[3]
                    dst = mouse_gray[y:y + height, x:x + width]
                    ret,img_threshold = cv2.threshold(dst,threshold,255,cv2.THRESH_BINARY)
                    #cv2.imshow("img_threshold",img_threshold)
                    cv2.imwrite("./output/img_threshold.jpg",img_threshold)
                    #print('savedimages.')
                    binary_img = cv2.imread("./output/img_threshold.jpg")
                    cnt =0
                    for val in binary_img.flat:
                        if val == 0:
                            cnt += 1
                    cv2.waitKey(1)
                    if cnt > 600:
                        #print(speaking_counter, "Speaking!!")
                        speaking_counter += 1
                    if speaking_counter == 10:
                        print("You spoke ", speaking_counter * 0.21, "second")
                        speaking_counter = 0
                    else:
                        continue
                    #print("done.")

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
