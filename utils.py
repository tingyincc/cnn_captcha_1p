
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


def hueChange(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv.shape[0]
    w = hsv.shape[1]
    random_hue = random.randint(0, 255)
    for y in range(0, h):
        for x in range(0, w):
            hsv[y][x][0] = (hsv[y][x][0] + random_hue) % 255

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def kmeans(img, K):
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

# segment: merge contours


def segmentDigit(image_path):
    image = cv2.imread(image_path)
    image = cv2.copyMakeBorder(
        image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    dst = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)
    dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(dst2, 5, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    rectList = []
    for i in range(1, len(contours)):  # len(contours)
        x, y, w, h = cv2.boundingRect(contours[i])
        rectList.append([x, y, w, h])

    rectList = sorted(rectList, key=lambda l: l[0])

    length = len(rectList)
    th_y = 20
    th_x = 4
    max_width = 25
    i = 0
    while i < length:
        j = 0
        while j < length:
            if(i == j):
                j += 1
                continue
            if(-th_x+rectList[i][0] < rectList[j][0] < th_x+rectList[i][0]+rectList[i][2] or -th_x+rectList[j][0] < rectList[i][0] < th_x+rectList[j][0]+rectList[j][2]):
                if(-th_y+rectList[i][1] < rectList[j][1] < th_y+rectList[i][1]+rectList[i][3] or -th_y+rectList[j][1] < rectList[i][1] < th_y+rectList[j][1]+rectList[j][3]):
                    if(max(rectList[i][0]+rectList[i][2], rectList[j][0]+rectList[j][2])-min(rectList[i][0], rectList[j][0]) > max_width):
                        print("pass")
                    else:
                        rectList.append([min(rectList[i][0], rectList[j][0]), min(rectList[i][1], rectList[j][1]), max(rectList[i][0]+rectList[i][2], rectList[j][0]+rectList[j][2])-min(
                            rectList[i][0], rectList[j][0]), max(rectList[i][1]+rectList[i][3], rectList[j][1]+rectList[j][3])-min(rectList[i][1], rectList[j][1])])
                        rectI = rectList[i]
                        rectJ = rectList[j]
                        rectList.remove(rectI)
                        rectList.remove(rectJ)
                        i = 0
                        j = 0
                        rectList = sorted(rectList, key=lambda l: l[0])
                        length -= 1
            j += 1
        i += 1

    rectList = sorted(rectList, key=lambda l: l[0])
    ret, thresh1 = cv2.threshold(dst2, 5, 255, cv2.THRESH_BINARY)
    digit_image = []
    for i in range(0, len(rectList)):  # len(contours)
        x, y, w, h = rectList[i][0], rectList[i][1], rectList[i][2], rectList[i][3]
        cv2.rectangle(thresh1, (x, y), (x+w, y+h), (0, 255, 0), 1)
        digit_image.append(image[y-3:y+h, x-3:x+w])
    return digit_image


# segment: color seperation
def segmentDigit_binary(image_path):
    image = cv2.imread(image_path)
    h = image.shape[0]
    w = image.shape[1]

    new_image = np.zeros((h, w, 3), dtype="uint8")
    final_binary = np.zeros((h, w), dtype="uint8")
    colorset = set()
    for y in range(1, h-1):
        for x in range(1, w-1):
            L = [image[y][x], image[y][x-1]]
            out = (np.diff(np.vstack(L).reshape(len(L), -1), axis=0) == 0).all()
            if(out):
                colorset.add((image[y][x][0], image[y][x][1], image[y][x][2]))
                new_image[y][x] = image[y][x]

    image_stack = []
    for color in colorset:
        singlecolorimage = np.zeros((h, w, 3), dtype="uint8")
        for y in range(1, h-1):
            for x in range(1, w-1):
                if((image[y][x] == color).all()):
                    singlecolorimage[y][x] = new_image[y][x]
        image_stack.append(singlecolorimage)

    rectList = []
    for im in image_stack:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        binary, contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            if(len(contours) > 1):
                break
            x, y, w, h = cv2.boundingRect(contours[i])
            if(600 > w*h > 150 and w < 30 and h < 30 and np.count_nonzero(im)/(w*h) > 0.5):
                rectList.append([x, y, w, h, np.count_nonzero(im)/(w*h*3)])
                final_binary = cv2.bitwise_or(final_binary, binary)

    rectList = sorted(rectList, key=lambda l: l[0])
    while(len(rectList) > 4):
        rectList = getOutlierIndex(rectList)

    digit_image = []
    for r in rectList:
        x, y, w, h = r[0], r[1], r[2], r[3]
        digit_image.append(cv2.resize(
            final_binary[y:y+h, x:x+w], (20, 20), interpolation=cv2.INTER_AREA))

    return digit_image


def getOutlierIndex(arr):
    upper = 12
    lower = 4
    for i in range(0, len(arr)-1):
        dist = arr[i+1][0] - (arr[i][0] + arr[i][2])
        if(dist > upper or dist < lower):
            if(i+2 >= len(arr)):
                arr.pop(i+1)
                return arr
            else:
                second_dist = arr[i+2][0] - (arr[i][0] + arr[i][2])
                # print(second_dist)
                if(lower < second_dist < upper):
                    arr.pop(i+1)
                    return arr
                else:
                    arr.pop(i)
                    return arr
    arr = sorted(arr, key=lambda l: l[4], reverse=True)
    arr.pop()
    arr = sorted(arr, key=lambda l: l[0])
    return arr
