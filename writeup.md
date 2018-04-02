###Model Description
In the encoder the network is using a Separable Convolution layer with 2x2 strides followed by a batch normalization layer. 
The 2x2 strides result in decreasing the first 2 dimensions in half.

This is the model architecture: 
1. Encoder: Separable Convolution layer with 32 filters resuting in (160x160x3) output. 

2. Encoder: Separable convolution layer with 64 filters resulting in (40x40x64) output. 

3. Encoder: Separable Convolution layer using 128 convolutions resulting in (20x20x128) output

4. One-by-one convolution with 128 filters which preserves the dimensions output is again (20x20x128)

5. Decoder: Bilinear upsampling with 128 filters which is concatenated with layer 2 of the encoder followed by 2 
separable convolutional layers with 128 filters each resulting in (40x40x128) output.

6. Decoder: Bilinear upsampling with 64 filters concatenated with layer 1 of the encoder followed by 2 separable 
convolutional  layer with 64 filters resulting in (80x80x64) output.

7. Decoder: Bilinear upsampling with 32 filters concatenated with the Input and followed by 2 separable conv layers
resulting in (160x160x32) output. 

8. Conv2D layer that converts the output to the desired num classes resulting in (160x160x3) output


##Model Summary
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 160, 160, 3)       0         
_________________________________________________________________
separable_conv2d_keras_1 (Se (None, 80, 80, 32)        155       
_________________________________________________________________
batch_normalization_1 (Batch (None, 80, 80, 32)        128       
_________________________________________________________________
separable_conv2d_keras_2 (Se (None, 40, 40, 64)        2400      
_________________________________________________________________
batch_normalization_2 (Batch (None, 40, 40, 64)        256       
_________________________________________________________________
separable_conv2d_keras_3 (Se (None, 20, 20, 128)       8896      
_________________________________________________________________
batch_normalization_3 (Batch (None, 20, 20, 128)       512       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 20, 20, 128)       16512     
_________________________________________________________________
batch_normalization_4 (Batch (None, 20, 20, 128)       512       
_________________________________________________________________
bilinear_up_sampling2d_1 (Bi (None, 40, 40, 128)       0         
_________________________________________________________________
concatenate_1 (Concatenate)  (None, 40, 40, 192)       0         
_________________________________________________________________
separable_conv2d_keras_4 (Se (None, 40, 40, 128)       26432     
_________________________________________________________________
batch_normalization_5 (Batch (None, 40, 40, 128)       512       
_________________________________________________________________
separable_conv2d_keras_5 (Se (None, 40, 40, 128)       17664     
_________________________________________________________________
batch_normalization_6 (Batch (None, 40, 40, 128)       512       
_________________________________________________________________
bilinear_up_sampling2d_2 (Bi (None, 80, 80, 128)       0         
_________________________________________________________________
concatenate_2 (Concatenate)  (None, 80, 80, 160)       0         
_________________________________________________________________
separable_conv2d_keras_6 (Se (None, 80, 80, 64)        11744     
_________________________________________________________________
batch_normalization_7 (Batch (None, 80, 80, 64)        256       
_________________________________________________________________
separable_conv2d_keras_7 (Se (None, 80, 80, 64)        4736      
_________________________________________________________________
batch_normalization_8 (Batch (None, 80, 80, 64)        256       
_________________________________________________________________
bilinear_up_sampling2d_3 (Bi (None, 160, 160, 64)      0         
_________________________________________________________________
concatenate_3 (Concatenate)  (None, 160, 160, 67)      0         
_________________________________________________________________
separable_conv2d_keras_8 (Se (None, 160, 160, 32)      2779      
_________________________________________________________________
batch_normalization_9 (Batch (None, 160, 160, 32)      128       
_________________________________________________________________
separable_conv2d_keras_9 (Se (None, 160, 160, 32)      1344      
_________________________________________________________________
batch_normalization_10 (Batc (None, 160, 160, 32)      128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 160, 160, 3)       99        
=================================================================
Total params: 95,961
Trainable params: 94,361
Non-trainable params: 1,600
```
Score:
```
Score Weight: 0.7469613259668508
Final IoU: 0.5735620790511294
Final Score: 0.42842869109233533
```



##Choice of Parameters
My initial model was only with 2 Encoder and Decoder layers and with a learning rate of 0.1. This quickly turned out to be producing bad results
Next I increased the number of layers and used 6,12,32 for the number of filters. I also reduced the learning rate to 0.001 and trained using a GPU.
This produced better results but my final score was around 0.3. I then increased the number of filters in each layer. 
I chose a batch size of 64 as to increase the speed of training on GPU without degrading the performance of the network.
I ran the network for 20 epochs which got me very close but was not enough so 25 produced the desired result. 
```
learning_rate = 0.001
batch_size = 64
num_epochs = 25
steps_per_epoch = 350
validation_steps = 50
```

##Separable convolution vs Regular convolution
The separable convolution is very similar to the regular convolution layer but its big advantage is that the number of parameters 
is greatly reduced. It is implemented by a convolution performed over each channel followed by 1x1 convolution which combines 
the outputs from the previous convolution layer. 

The resulting reduction in parameters is greatly beneficial for performance and very useful for mobile devices.

##1x1 Convolution
The 1x1 Convolution is very similar to the fully connected network but with the benefit of preserving spatial information and dimensionality.
It is used in the network to connect the encoder and decoder and to convert the final output to the desired number of classes.

The fully connected layer is usually used for tasks that don't require the output to be an image. It is better for object 
classification and recognition. 

##Benefits and drawbacks
The Fully convolutional network has the benefit of describing and categorizing each pixel in the image providing for a much greater detail. 
This can be also a drawback when we are trying to detect larger objects and entities and their positions. In cases where we want to detect a 
generic object and its position in the scene the YOLO network would be better to use. 

##Performance on different data
The network is performing well for human subjects but will need further tweaking for more categories(cats,dogs, cars) we will probably need much more training data
and much more complicated network as the number of classifications increase.  

##Conclusion
This is the second time that I am doing a fully convolutional neural network. My previous attempt was while going to the 
SDCND. The network architecture has changes since then by using the separable convolution, this simplifies the design and implementation
without impacting the performance. 
