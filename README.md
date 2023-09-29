# Gender and Age Detection using PyTorch CNN
## This was created as my university project, I had to detect the person's gender and age using the image that can be given to the model to predict.
The dataset I used is the UTKFace dataset that has 23708 images with male and female genders, ages varying from 1-116 and 4 ethnicities.\
The ages are scaled using Sklearn's MinMaxScaler to scale the ages to 0-5.\
For the gender I used classification, Cross Entropy Loss.\
And for the age I used Regression with SmoothL1Loss.\
The gender accuracy on test set gives the accuracy of 96-98%, whereas the age accuracy gives the mean absolute error of 0.2-0.3. 

### I added an ArgParser in the main.py file to add filepath as an argument and -acc argument to check for accuracy.
