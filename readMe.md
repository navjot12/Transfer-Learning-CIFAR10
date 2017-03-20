# Transfer Learning on CIFAR10 dataset

Transfer Learning uses information gained by training of a model on one set of data, to make training/testing on some other set of (similar) data more efficient.  

This python script uses Keras module based on TensorFlow backend to train for categorization of CIFAR10 image dataset by using Transfer Learning. The script uses two models- first model trains for categorization of images in first 5 classes in the CIFAR10 dataset; the second (trans) model uses the information gained by the first model to train for categorization of images in the next 5 classes in CIFAR10.

#### Dataset
The CIFAR10 dataset is a collection of 50000 32x32 coloured images belonging to one of 10 classes. 25000 images belong to first 5 classes - 0..4; the rest of 25000 to classes 5..9. 80% of these datasets were used for training under the respective model, while the other 20% was used for validation of the respective model.

#### Models

The convolutional neural network for gaining information by training a model for first 5 classes is as follows:  
	Layer 1. Convolutional layer with 32 5x5 kernels, expecting input of shape (32, 32, 3) and using relu as activation function.  
	Layer 2. Second Convolutional layer with 16 5x5 kernels, using relu as activation function.  
	Layer 3. MaxPooling layer which pools the (24,24,16) shaped output from previous layer in (2,2) pools, rendering an output of (12,12,16).  
	Layer 4. Third Convolutional layer with 8 3x3 kernels, using relu as activation function.  
	Layer 5. Flatten layer which flattens the (10,10,8) output from previous layer into a single vector of shape (800,).  
	Layer 6. Dropout layer with rate 0.42.  
	Layer 7. Fully connected layer with 128 neurons, using relu as activation function.  
	Layer 8. Fully connected layer with 5 neurons - acting as the final output classes.  

The convolutional neural network for training a model for classes 5..9 by using the information gained by previous model is as follows:  
	Layer 1-6: First 6 layers of the previous model (basically the convoluted layers) are used as it is in this model. Further they are made non-trainable to preserve the information gained by them while training of previous model.  
	Layer 7: Fully connected layer with 128 neurons, using relu as activation function.  
	Layer 8. Fully connected layer with 10 neurons - acting as the final output classes.  

### Result Summary:
The output of the python script can be found in the **results.txt** file.  

The two models were trained on a CPU for just 10 epochs each.  

The first model trained for classes 0..4, achieved a validation accuracy of *69.03%* after 10 epochs. However, the training time was **16 minutes and 14 seconds**.  

The second model trained for classes 5..9 used transfer learning to expedite the training process. It achieved an accuracy of **76.06%** after 10 epochs. More importantly the training time was down to just **6 minnutes and 49 seconds**.  

**Therefore, transfer learning significantly brought down the time for processing of data**.
