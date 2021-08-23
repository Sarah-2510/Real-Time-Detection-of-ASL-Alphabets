# Real-Time-Detection-of-ASL-Alphabets

![A1](https://user-images.githubusercontent.com/74179721/128869014-05dced21-233a-40ad-b2f6-a3f353cacaea.jpg)

## Data
### Link to the dataset - https://www.kaggle.com/grassknoted/asl-alphabet 
The dataset was taken from kaggle. 

### Data Description
The ASL data set is a collection of images of alphabets from the American Sign Language. It is divided
into training and testing datasets.The training data set contains 87,000 images. There are 29 classes, 
of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING. Each class
has about 3000 images.The test dataset contains sample images for each alphabet. 

## Work Flow 

### 1. Pre-Processing

In total there were 43,500 images. To read and pre-process images, CV2 was used. Since the data was large , training and testing would take a lot of time, therefore only a subset of 1500 images were used. 

### 2. Creating the data.csv file.

In this section , a data.csv file was created. It maps the image paths to target class. It is divided into two columns. The first is is image_path which will hold the image paths. The second column  is the target column  which indicates the class of images(0 to 28). 

### 3. Writing our neural network architecture inside cnn_models.py file.

In this , a customized CNN model was created. It has four 2D convulational layers(self.conv1.. self.conv4) , 2 linear layers(self.fc1 & self.fc2) and a max pool layer(self.pool).In the forward function , max-pooling is applied to the activations of every convulational layer. 

### 4. Training the CNN model 
* The CNN model is being trained on the pre-processed images. 

* Two functions are being used : 

  i) fit - for training the model on the train dataset 

  ii) validation - for checking the models performance

* These functions compute and return the loss and accuracy of training and validation dataset on the model. On each epoch these parameters are appended to lists so, that they can be plotted and visualized. The accuracy and loss plots are plotted using matplotlib 

### 5. Testing the CNN model
* The image that has to be tested is loaded using cv2 package,it is resized and preprocessed to match the format of the images in the test dataset

* Writing the test code.
Then a code was written to detect the sign language letters inside  cam_test.py file for real-time
webcam feed.

* The image is then provided to the model and final predictions are made

## Result 

* The model is predicts the alphabets correctly . 

![Picture3](https://user-images.githubusercontent.com/74179721/130441622-b8fa1e32-9b48-4456-ae98-222980ddea12.png)

* Model Performance Summary 

i)The final validation accuracy is 96.99 and train accuracy is 98.93


![Picture1](https://user-images.githubusercontent.com/74179721/130441642-05c55ecc-3a0d-43ac-a004-cb27bfe2c944.png)

ii)The final validation loss is 0.0046 and the train loss is 0.0012. 

![Picture2](https://user-images.githubusercontent.com/74179721/130441632-d8260acd-153b-425e-845e-8ac39204fd4b.png)


