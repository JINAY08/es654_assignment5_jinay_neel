## Comparison amongst the models is shown below:

<p align = "center">
  
   | **Model** | **Training Time** | **Training Loss** |  **Training Accuracy** | **Testing Accuracy** | **Number of Parameters** |
   |:-----------:|:-----------------:|:----------------:|:----------------------:| :-------------------:| :-----------------------:|
   |   VGG (1 block)        |      54.0295     |    0.5503   |     68.75      |      64.99     |   40961153    | 
   |   VGG (3 block)        |      132.503     |    0.5704   |     75     |      67.5     |   10333505    | 
   |   VGG (3 block with Data Augmentation)        |      319.721     |    0.51635   |     75.625     |      67.5     |   10333505    | 
   |   Transfer Learning (10 epochs)       |     427.35    |    0.09545   |     98.75     |      95     |   17926209    |
   |   User-defined MLP        |      117.78     |    0.433   |     83.75     |      57.499     |   133374977    | 
</p>

## Tensorboard Visualization:

### VGG 1 Block Model:

![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_1block_accuracy.png)
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_1block_loss.png)
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_1block_images.png)


### VGG 3 Block Model:

![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_3block_accuracy.png)
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_3block_loss.png)
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_3block_images.png)

### VGG 3 Block Model (Data Augmentation):

![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_3blockdataaugment_accuracy.png)
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_3block_dataaugment_loss.png)
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/vgg_3block_dataaugment_images.png)

### Transfer Learning Model:
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/transfer_test_accuracy.png) ![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/transfer_train_accuracy.png)
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/transfer_test_loss.png)![Plot]( https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/transfer_train_loss.png))
![Plot](https://github.com/JINAY08/es654_assignment5_jinay_neel/blob/main/images/transfer_images.png)

## Various Insights:
> Are the results as expected? Why or why not?
- The results show that the VGG model with three blocks (without and with data augmentation) tends to have higher training accuracy and slightly better testing accuracy compared to the VGG model with only one block. This is expected as deeper models can capture more complex features in the data. However, the training time for VGG models with more blocks is also significantly higher, which is also expected as deeper models require more computation. The transfer learning model performs exceptionally well with high training accuracy and testing accuracy, which is also expected as transfer learning leverages pre-trained models to benefit from their learned features.
  
> Does data augmentation help? Why or why not?
- The VGG model with 3 blocks and data augmentation shows comparable training accuracy and similar testing accuracy as the VGG model with 3 blocks without data augmentation. This suggests that data augmentation may not have a significant impact on the performance of VGG models in this case, possibly due to the small size of the dataset.
  
> Does it matter how many epochs you fine tune the model? Why or why not?
- The number of epochs for fine-tuning the model has an impact on the model's performance. It is important to note that too few or too many epochs negatively impact the model's performance. Too few epochs results in underfitting, while too many epochs results in overfitting. The optimal number of epochs for fine-tuning depends on the dataset and the complexity of the model.

> Are there any particular images that the model is confused about? Why or why not?
- Based on the images from the test set and their predictions, there are many images that the model is confused about. This could be due to various reasons such as the images having similar features or patterns from both classes, low image quality, or lack of diversity in the training data. Fine-tuning the model with more diverse data, or using techniques such as regularization, could help in improving the model's performance on such images.

> What can you conclude from the MLP model performance?
- The user-defined MLP has a significantly larger number of parameters compared to VGG models and performs better in terms of training accuracy, but the testing accuracy is lesser than VGG models. This suggests that while a larger MLP model can fit the training data better, it may not necessarily generalize well to unseen data, as indicated by the testing accuracy. This highlights the importance of model architecture and not just the number of parameters in determining model performance.
