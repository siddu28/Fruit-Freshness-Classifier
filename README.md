# Fruit-Freshness-Classifier
# Problem statement 
This project focused on building a robust computer vision model to classify whether a fruit is fresh or rotten using image data. Here's a detailed breakdown of the steps I followed:

🧾 1. Data Loading & Preprocessing
Loaded the dataset and ensured all images had a uniform size, which is crucial for CNNs to perform well.

Applied image transformations like rotations, horizontal flips, and scaling.

Converted images to tensors and normalized them using specific mean and standard deviation values.

🔁 I used data augmentation to artificially increase the diversity of the dataset. This helps the model generalize better by introducing variations like rotated or flipped images – especially helpful when the dataset size is small.

🧪 2. Data Batching
Divided the training data into batches of 32 images.

This speeds up the training process and requires less memory.

Though batching introduces some noise, it helps the model converge faster. The Adam Optimizer played a key role in smoothing out these variations due to its adaptive learning rate and momentum.

🧠 3. CNN Model Training & Validation
Implemented a Convolutional Neural Network (CNN) from scratch:

Applied Convolution layers to extract features from RGB images.

Used ReLU activation for non-linearity.

Followed with MaxPooling to retain dominant features while reducing dimensionality.

Repeated the above process with increasing filters to capture higher-level features.

Trained the model on the training dataset and validated it on the validation set.

🛡️ 4. Regularization
Used Batch Normalization to stabilize and speed up training.

Helped in preventing overfitting and improved model accuracy.

🧠🔁 5. Transfer Learning with ResNet50
Leveraged Transfer Learning using a pretrained ResNet50 model.

Froze the convolutional base (already trained on millions of images via ImageNet).

Modified and trained the final fully connected layer to suit the fruit classification task.

Achieved ~98% accuracy on the validation set! 🎯

![Image](https://github.com/user-attachments/assets/aff71a3e-84be-4012-87c0-3a5dd53d9538)

![Image](https://github.com/user-attachments/assets/5cb20f82-5758-4ed9-87b0-70f0a3592790)


# Streamlit Demo - https://fruit-freshness-classifier.streamlit.app/


### Set Up

1. To get started, first install the dependencies using:
    ```commandline
     pip install -r requirements.txt
    ```
   
2. Run the streamlit app:
   ```commandline
   streamlit run app.py
