
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import csv
from utils import kmeans, segmentDigit, segmentDigit_binary
from matplotlib import pyplot as plt

LETTERSTR = "123456789ABCDEFGHJKLMNPQRSTUVWXYZefqw"
singleDetect = False

model = load_model("./res_model/model.h5")


def image_test(path):
    test_image = []
    digit_images = segmentDigit(fullpath)
    for i in digit_images:
        #i = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        i = cv2.resize(i, (25, 30), interpolation=cv2.INTER_CUBIC)
        if(singleDetect):
            plt.imshow(i)
            plt.show()
        nparr = np.array(i)
        #nparr = np.expand_dims(nparr, axis=2)
        nparr = nparr / 255.0
        test_image.append(nparr)
        print(nparr.shape)
    test_image = np.stack(test_image)
    print(test_image.shape)

    prediction = model.predict(test_image)
    answer = ""
    for predict in prediction:
        answer += LETTERSTR[np.argmax(predict)]

    return(answer)


def image_test_color_seg(path):
    test_image = []
    digit_images = segmentDigit_binary(fullpath)
    for i in digit_images:
        if(singleDetect):
            plt.imshow(i)
            plt.show()
        nparr = np.array(i)  # 轉成np array
        nparr = np.expand_dims(nparr, axis=2)
        nparr = nparr / 255.0
        test_image.append(nparr)
    test_image = np.stack(test_image)

    prediction = model.predict(test_image)
    answer = ""
    for predict in prediction:
        answer += LETTERSTR[np.argmax(predict)]

    return(answer)


if(singleDetect):
    fullpath = "./verify_img/verify_113.png"
    ans = image_test_color_seg(fullpath)
    print(ans)
else:
    correct = 0
    total = 0
    with open('captcha_test.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            total += 1
            fullpath = "./verify_img/"+row[0]+".png"
            ans = image_test_color_seg(fullpath)

            if(ans == row[1]):
                correct += 1
            else:
                print("Wrong Predict: ", newline="")
            print(
                f"GT: {row[1]}, Predict: {ans}. Accuracy:{correct/total}, {correct}/{total}")
        # with open("result.csv", 'a+', newline='') as f:
        #     csv_write = csv.writer(f)
        #     csv_head = [row[0], ans]
        #     csv_write.writerow(csv_head)
