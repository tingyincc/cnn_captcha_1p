# cnn_captcha_1p

CNN model for 1p3a captcha solving using small dataset (100 captcha images).

Model Accuracy : 100% on  268/111 training/validation sets.

Test Accuracy：100% on 90 test images. 

## Dependency
|  Package| Version |
|--|--|
| requests| 2.23.0 |
| numpy | 1.18.1 |
| matplotlib | 3.1.3 |
|tensorflow|2.0.0|
|opencv|3.4.2|


## Usage

1. `python creat_training_sets.py` to slice a captcha image to four single character images
2. prepare your labeled `captcha_train.csv`and then `python train.py`
3. `python test.py` will compare results with `captcha_test.csv` 
	
|  File name| Description |
|--|--|
| create_training_sets.py| convert a captcha image to four single character images |
| train.py | create model and train |
| test.py | load model and test |
|utils.py|image processing and segmentation|

|  Folder name| Description |
|--|--|
| res_model | trained model |
| verify_img | original captcha images. (train: 0-69 valid: 70-99 test: 100-189) |
| train_data | segmented images for training|
| valid_data | segmented images for validation|

## Methods

### Preparing Training Set

Collect 100 images from target website. 
Refer: function crack_verify in auto_1p3a.py to save the image:
```python
res = self.session.get(url)
with  open("img.jpg", 'wb') as fp:
	fp.write(res.content)
```
Or use selenium to crop the screen shot.

### Preprocessing
1. Observe the image carefully:

![enter image description here](https://upload.cc/i1/2020/06/12/RPzto6.png)

We can see each charater is composed by a same color.

2. We use the set to record all colors without repeat. Note that we only put the color of the pixel which has dajacent same color pixel on it's right to eliminate the backgroud noise (because the noise barely has same color on the neighbor). 
We put a single color image into a stack.

![enter image description here](https://upload.cc/i1/2020/06/12/ZCAVuI.jpg)

Now we have 12 color seperate images in the stack. 

![enter image description here](https://upload.cc/i1/2020/06/12/0puhgv.jpg)

3. We use `cv2.findContours` for each image. If the contour fits our criteria, where number of contour == 1, area (w*h) between 150-600, w<30, h<30, we will record its x,y,w,h, and color pixel percentage in a rectangle list.

![enter image description here](https://upload.cc/i1/2020/06/12/vGlgdN.jpg)

4. Then we turn all images to a binary image

![enter image description here](https://upload.cc/i1/2020/06/12/OdCVoy.jpg)

5. But there still an outlier exist. Thus, we use other criterias to get rid of it. First, we sort the rectangle list by x, then we get the distance between the rectangles(boxes). All the distance should between 4-12.
In this case, all the rectangles passed this criteria.

![enter image description here](https://upload.cc/i1/2020/06/12/aN2FZi.jpg)

However, for example, if the boxes look like the image below, 4 and the outlier overlaped because of distance -11px. Later, we decide which one has a normal distance with B, then we keep it, which in here is 4.

![enter image description here](https://upload.cc/i1/2020/06/12/0bmtHl.jpg)


6. If the boxes pass the first criteria, we choose the outlier with lowest percentage of color pixels/whole pixels. 

![enter image description here](https://upload.cc/i1/2020/06/12/mvbNun.jpg)

7. Now we have 4 correct rectangles! We resize them to the same size and save to each folders.

![enter image description here](https://upload.cc/i1/2020/06/12/Arp7VK.jpg) 

![enter image description here](https://upload.cc/i1/2020/06/12/Q1jY2x.png)

After resizing, they looks similar:

![enter image description here](https://upload.cc/i1/2020/06/12/ws31jx.png)

### Training
We use the model and the structure from here :
[https://github.com/JasonLiTW/simple-railway-captcha-solver/blob/master/README.md](https://github.com/JasonLiTW/simple-railway-captcha-solver/blob/master/README.md)
with batch size=40, epochs=300, and train data shuffling.
And the output layer was modified to single digit ouput. 

  
```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 20, 20, 1)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 20, 20, 32)        320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 18, 18, 32)        9248
_________________________________________________________________
batch_normalization (BatchNo (None, 18, 18, 32)        128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 9, 9, 32)          0
_________________________________________________________________
dropout (Dropout)            (None, 9, 9, 32)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 9, 64)          18496
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 64)          36928
_________________________________________________________________
batch_normalization_1 (Batch (None, 7, 7, 64)          256
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 3, 64)          0
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 3, 64)          0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 3, 128)         73856
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 1, 128)         147584
_________________________________________________________________
batch_normalization_2 (Batch (None, 1, 1, 128)         512
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 1, 128)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 1, 128)         0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 1, 256)         33024
_________________________________________________________________
batch_normalization_3 (Batch (None, 1, 1, 256)         1024
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 1, 1, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 256)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0
_________________________________________________________________
digit (Dense)                (None, 37)                9509
=================================================================
Total params: 330,885
Trainable params: 329,925
Non-trainable params: 960
```

Note: I manually pick some cases into the train folder to let data more balance. If you generate train and valid data from verify_img, you'll find the number of files are different.

Approximately after 150 epochs, acc and val_acc will be 100%.

![enter image description here](https://upload.cc/i1/2020/06/16/SdfrFo.png)

Accuracy: (*Orange- Train  Blue-Validation*)

![enter image description here](https://upload.cc/i1/2020/06/16/V9DxvI.png)

Loss:  

![enter image description here](https://upload.cc/i1/2020/06/16/O5mnPx.png)


### Testing

Using no.100-189 for testing. 

## Experiements

### Without Preprocessing : 4 digits
If we use the simple railway captcha solver for 4 digits, the 100 images are not enough for training:

     - 0s - loss: 0.0524 - digit1_loss: 0.0088 - digit2_loss: 0.0215 - digit3_loss: 0.0046 - digit4_loss: 0.0176 - 
    digit1_acc: 1.0000 - digit2_acc: 1.0000 - digit3_acc: 1.0000 - digit4_acc: 1.0000 - 
    val_loss: 31.2516 - val_digit1_loss: 7.0900 - val_digit2_loss: 8.6135 - val_digit3_loss: 7.3375 - val_digit4_loss: 8.2105 - 
    val_digit1_acc: 0.1333 - val_digit2_acc: 0.0667 - val_digit3_acc: 0.0667 - val_digit4_acc: 0.0667

### Preprocessing using RGB : single digit
Therefore, I tried to segment one image to four character images, and the result improved: 
	
	Epoch 100/100
	- 1s - loss: 0.1641 - acc: 0.9438 - val_loss: 0.5276 - val_acc: 0.9038

### Preprocessing using RGB and data augmentation : single digit
Then I tried the data augmentation for hue change (as `function hueChange(image)` in utils.py), it improved a little bit :

    Epoch 50/50
    - 165s 620ms/step - loss: 1.4040e-04 - acc: 1.0000 - val_loss: 0.4662 - val_acc: 0.9712

However, since the improvement is unsignificant and extend the training time, I decided to turn these color images to gray images. 

## Todo
- Online (website) test
- Super parameter tuning to reduce epochs
- Model validation (I am not sure whether my model design correctly)
- Using pixel comparing instead deep learning (since single character image looks so similar...)

