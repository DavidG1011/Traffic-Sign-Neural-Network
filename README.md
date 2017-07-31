**Traffic Sign Recognition Writeup** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./New_Images/writeup/gray.png "Grayscaled Image"
[image2]: ./New_Images/writeup/traingraph.png "Training Graph"
[image3]: ./New_Images/writeup/validgraph.png "Validation Graph"
[image4]: ./New_Images/writeup/validgraph.png "Testing Graph"
[image5]: ./New_Images/1.jpg "Image 1"
[image6]: ./New_Images/4.jpg "Image 2"
[image7]: ./New_Images/8.jpg "Image 3"
[image8]: ./New_Images/13.jpg "Image 4"
[image9]: ./New_Images/14.jpg "Image 5"
[image10]: ./New_Images/stop32.png "Stop 32x32"
[image11]: ./New_Images/yield32.png "Yield 32x32"



**Rubric Points:**

The rubric covered [Here](https://review.udacity.com/#!/rubrics/481/view) was the basis for the approach to this project. 

**1. Preprocessing:**
In order to reduce the amount of image data and ease the process of training, I used the preprocessing methods of grayscaling and normalization. I used these methods for the simplicity and effectiveness of them.

**Gray:**

![alt text][image1]


**Normalization:**

Normalization was as simple as taking the pixel values of each image and applying the formula of (pixel - 128) / 128. This reduces pixel data without changing the content of an image.


---

***Data Set Summary & Exploration***

I used the numpy library and built in python len() to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

***Visual Exploration***

I created bar graphs of my data to show the amount of signs per class for each of the data sets. I chose bar graphs as they are easy to read, while still providing a helpful visual representation of the size of my data set.

***Training Graph***

![alt text][image2]

***Validation Graph***

![alt text][image3]

***Testing Graph***

![alt text][image4]



***Design and Test a Model Architecture***


The architecture I used was a modified LeNet 5 - [Link](http://yann.lecun.com/exdb/lenet/). I feel that it is perhaps too simplistic a model, but it still sports a relatively high accuracy with some added touches.

The final model I decided on was:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray, Normalized image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x16     									|
| RELU		|        									|
| Max pooling				| 2x2 stride, outputs 5x5x16        									|
|	Flatten				|		Outputs 400										|
|	Fully Connected Conv:		|		In 400, Out 120						|
| RELU |                                   |
| Dropout Layer | Probability 0.5 |
| Convolution | In 120, Out 84 |
| RELU |                      |
| Dropout Layer | Probability 0.5 | 
| Convolution | In 84, Out 43: (Number of Classes) | 
 

**Training The Model**

To train the model, I used an Adam Optimizer from the tensorflow library. I found this was an adequate optimizer with the correct learning rate. My findings from doing testing were that the learning rate was the most crucial paramater to the accuracy of my model.

The generalized results are as follows:


| Learning Rate    	|  Validation Accuracy        					| 
|:-----------------:|:--------------------------:| 
| > 0.01| Very poor. Usually stuck at less than 1%|
| < 0.01| Good. Diminishing returns if too low. | 


The final paramaters I landed on are as follows:

* Epochs = 100    

May be a bit high, results tend to stagnate towards the end.

* Batch Size  = 1000

Quick with a 2GB NVIDIA GPU. 

* Learn Rate = 0.002 

The lucky number I came to after many tunings.

My final model results were:
* validation set accuracy of 96.281%
* test set accuracy of 94.3%
---

***Using an Iterative Model to Improve:***

* The first model I used was a straight LeNet5 with no modifications, to achieve a baseline of how well it would perform with the dataset. This was also before any image normalization. 
* The problem I ran into with this was that the model was constantly at a low accuracy. It wouldn't budge past under 10% accuracy or so. Tuning all the paramaters, I could only achieve at most a very low validation rate of under 50%.
* From here, I modified the model to use Grayed and Normalized images.
* Next, I retested my model on the validation set only to notice a still low validation accuracy. 
* From here I realized that my model was overfitting. To counter this, I added two dropout layers with a 50% keep probability. Tuning my models learning rate, I finally achieved an acceptable accuracy of 96.281%

***Testing The Model On New Images***


The 5 new images I used for testing:

---


![alt text][image5]

![alt text][image6] 

![alt text][image7] 

![alt text][image8] 

![alt text][image9]


---

***Image Quality: Here Comes The Fuzz:***

A difficulty I faced using these images is that some are quite big, even when cropped from their original size. Cropping these large images down further to only 32x32 has made them very compressed, and possibly made it hard for the network to discern characteristics for these specific images.

Comparing these images to their originals above, it can be seen that resizing these images vastly changes the data of it.

---


![alt text][image10]

![alt text][image11]


---

***How Resizing to 32x32 Changes An Image Numerically***


| Original Image Size (pixels)		        |   Resize Pixel Loss X   | Resize Pixel Loss Y |
|:---------------------:|:-------------------:|:--------------------------:| 
| Speed limit (30km/h) - 122 x 119     		| ~73.77%							| ~73.11%         |
| Yield  - 297x 263 		                  	| ~89.23%							| ~87.83%         |
| Stop - 498 x 455				                   | ~93.57%							| ~92.97%         | 
| Speed limit (70km/h) - 	56 x 52      		| ~42.86%				 	 | ~38.46%         |
| Speed limit (120km/h) - 119 x 100			   | ~73.11%   		  | ~68.0%          |



Looking at these losses, it's easy to see how the network may have difficulty identifying these images. For example, the stop sign image lost over 93% of its original X pixel value and almost 93% of its original Y pixel value. These high loss resized images could contribute to my findings for prediction accuracy...

---

***Image Qualities***

**Resolution/Distortion:** 
As mentioned above, resizing the images reduces the resolution or overall image data to a very small percentage. This is probably the most limiting factor when it comes to predicting these images.

**Brightness:**
Good, very bright. These images were also normalized with the same method as the other image sets.

**Viewing Angle:**
The angle that the signs are at should be fine for prediction. They are relatively forward facing. 

**Edges:**
I made sure to crop the images as best as possible to reduce outer edge noise, some extra image data is left, however. 

---

***The results I got from my predctions were as follows:***


| Image			        |     Prediction	        					| Correct |
|:---------------------:|:-------------------:|:--------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h) 									| Yes|
| Yield     			| Yield										| Yes|
| Speed limit (30km/h)				| Stop										| No| 
| Speed limit (30km/h)	      		| Speed limit (70km/h)					 				| No|
|  Speed limit (120km/h)			| Speed limit (120km/h)     							| Yes|



***The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%***

---


**Comparing These Results To Test Set Results***


| Test Set Accuracy		 | Prediction Accuracy | Difference |
|:-------------------:|:--------------------:|:---------:| 
| 94.3%               | 60%                  |  - 34.3%  |



Comparing to my test set accuracy, it performed unfavorably with a 34.3% difference in accuracy. I feel this could be due to how lossy the images were once compressed down to 32x32. If I were to select images again, I would try to find ones with a smaller original size, to reduce the amount of compression. Using different resizing methods in python and using methods such as antialiasing seemed to make no difference in accuracy.


This is not as well as I had hoped the model would do, but in viewing the probability distribution for the model, the correct prediction is within the top 5 probabilities of each wrong image.

---


***The probabilities for each new sign are as follows:***

---

***Image 1: Speed limit (30km/h)***


| Prediction        	|     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (30km/h)   |        100.0000000000%|
|Speed limit (50km/h)     |        0.0000000000%|
|Speed limit (70km/h)       |      0.0000000000%|
|Stop                       |      0.0000000000%|
|Speed limit (60km/h)       |      0.0000000000%|

---

***Image 2: Yield***


| Prediction        	|     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
|Yield                    |      100.0000000000%|
|Speed limit (20km/h)      |       0.0000000000%|
|Speed limit (30km/h)       |      0.0000000000%|
|Speed limit (50km/h)        |     0.0000000000%|
|Speed limit (60km/h)         |    0.0000000000%|

---

***Image 3: Stop***


| Prediction        	|     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (30km/h)   |         99.9953508377%|
|Stop                    |         0.0033806267%|
|Yield                    |        0.0012675282%|
|Speed limit (50km/h)      |       0.0000002531%|
|Keep left                  |      0.0000000453%|

---

***Image 4: Speed limit (70km/h)***


| Prediction        	|     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (30km/h)   |         99.9999403954%|
|Speed limit (60km/h)    |         0.0000351977%|
|Stop                     |        0.0000248258%|
|Speed limit (70km/h)      |       0.0000027374%|
|Turn right ahead           |      0.0000000031%|

---

***Image 5: Speed limit (120km/h)***


| Prediction        	|     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (120km/h)     |      73.9777505398%|
|Speed limit (20km/h)       |     22.8198796511%|
|Speed limit (30km/h)        |     2.8353558853%|
|Go straight or right         |    0.2367284149%|
|Keep right                    |   0.0944264291%|

---

***Conclusion:***

***Overall I am pleased with the testing and validation accuracy. I am a little suprised with my new images not doing as well, but the features are being recognized, as evidenced by the probabilities of the correct sign being in the top 5 results. This was a cool exploration into the world of neural networks, and I hope to use what I learned here to make even better models in the future.***
