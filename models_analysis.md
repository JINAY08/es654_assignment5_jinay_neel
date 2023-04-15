Comparison amongst the four models is shown below:

<p align = "center">
  
   | **Model** | **Training Time** | **Training Loss** |  **Training Accuracy** | **Testing Accuracy** | **Number of Parameters** |
   |:-----------:|:-----------------:|:----------------:|:----------------------:| :-------------------:| :-----------------------:|
   |   VGG (1 block)        |      54.0295     |    0.5503   |     68.75      |      64.99     |   40961153    | 
   |   VGG (3 block)        |      132.503     |    0.5704   |     75     |      67.5     |   10333505    | 
   |   VGG (3 block with Data Augmentation)        |      319.721     |    0.51635   |     75.625     |      67.5     |   10333505    |   
   |   User-defined MLP        |      81.385     |    0.3928   |     81.875     |      67.5     |   287721473    | 
</p>

- Are the results as expected? Why or why not?
The results show that the VGG model with three blocks tends to have higher training accuracy and slightly better testing accuracy compared to the VGG model with only     one block. This is expected as deeper models can capture more complex features in the data. However, the training time for VGG models with more blocks is also           significantly higher, which is also expected as deeper models require more computation.
  
- Does data augmentation help? Why or why not?
The VGG model with 3 blocks and data augmentation shows similar testing accuracy as the VGG model with 3 blocks without data augmentation. This suggests that data       augmentation may not have a significant impact on the performance of VGG models in this case, possibly due to the small size of the dataset.
  
- Does it matter how many epochs you fine tune the model? Why or why not?
  

- Are there any particular images that the model is confused about? Why or why not?


- What can you conclude from the MLP model performance?
The user-defined MLP with a significantly larger number of parameters compared to VGG models performs better in terms of training accuracy, but the testing accuracy is   similar to VGG models. This suggests that while a larger MLP model can fit the training data better, it may not necessarily generalize well to unseen data, as           indicated by the testing accuracy. This highlights the importance of model architecture and not just the number of parameters in determining model performance.
