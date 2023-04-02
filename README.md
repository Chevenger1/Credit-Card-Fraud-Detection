# Credit-Card-Fraud-Detection
This project is one of my submission on Kaggle competition .
This project aimed to develop a credit card fraud detection system using a deep learning network. The dataset used for this project contained transaction records of credit card transactions, including both fraudulent and non-fraudulent transactions.

The project started with data exploration, where we gained insights into the characteristics of the data. We found that the data was highly imbalanced, with fraudulent transactions accounting for less than 1% of the total transactions. This presented a challenge, as traditional classification models tend to struggle with highly imbalanced datasets.

To overcome this challenge, we applied various techniques to balance the dataset, such as oversampling the minority class using SMOTE (Synthetic Minority Over-sampling Technique) and undersampling the majority class. We also used stratified sampling during the train-test split to ensure that both the training and testing sets had a representative distribution of fraudulent and non-fraudulent transactions.

We then developed a deep learning model using Pytorch, which consisted of several dense layers with varying numbers of neurons and activation functions. We used binary cross-entropy as the loss function and Adam as the optimizer.

After training and testing the model, we achieved a high accuracy and F1-score, with minimal false negatives (fraudulent transactions incorrectly labeled as non-fraudulent). Overall, the project successfully developed a credit card fraud detection system using a deep learning network, while also addressing the challenge of working with highly imbalanced data.
