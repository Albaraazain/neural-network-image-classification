import os
import pandas as pd

import cv2
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder


# Load the dataset from a directory
def load_dataset(base_dir):
    data = []
    # Walk through the directory with their subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # checking for images
            if file.endswith(".jpg") or file.endswith(".png"):
                # We get the class label from the folder name
                label = os.path.basename(root)
                # Collect the path to the file and its label
                data.append((os.path.join(root, file), label))

    # Convert the list of tuples into a pandas DataFrame with columns image_path and label so we can access them later as fields.
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    return df


# Function to compute histogram features for image classification
def hist_features(image_paths, bins=256): # we get a list of image paths and we return the features of each image all in an list
    # List to hold histograms for all images
    features = []

    # Process each image path in the list
    for path in image_paths:
        # First we read the image in greyscale
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Skip processing if the image cant be loaded
        if image is None:
            continue

        # calculate the histogram with the specified number of bins
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        # Normalizing the histogram so the sum of all bin values is 1
        hist = cv2.normalize(hist, hist).flatten()

        # Append the normalized histogram to the features list
        features.append(hist)

    # we Convert the list of histograms to a numpy array for easier handling in machine learning models
    feature_matrix = np.array(features)
    return feature_matrix


# function to extract LBP features which is good for texture classification
def mystery_features(image_paths, P=8, R=1):
    features = []  # list where wll store the LBP histograms for each img

    for path in image_paths:
        # First we need to read in the image from the path
        image = imread(path)
        # Then we convert the image to grayscale because LBP works on single channel images
        image_gray = rgb2gray(image)


        # Scaling pixel values to the 0-255 range since the LBP function expects this kind of input, it doesnt work well with floating point number i got a warning
        image_gray = (image_gray * 255).astype('uint8')

        #now we get the LBP of the image using the parameters P and R
        lbp = local_binary_pattern(image_gray, P, R, method="uniform")
        # We need a histogram of the LBP results and it has to include every possible value
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P * R + 3), range=(0, P * R + 2))

        # now we need to normalize which is important  to make the histogram a valid probability distribution
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        # we then add each histogram to our list of features
        features.append(hist)

    # Convert the list to a numpy array because its better and easer to work with. numpy allow parallel computing.
    return np.array(features)


# Why Local Binary Patterns (LBP)? I chose Local Binary Patterns as a feature type because if we want to
# differentiate between images like buildings, forests, and glaciers, it would be easy to just look at their texture
# in order to classify an image into one of those categories. plus LBP is robust against changes in lightning and
# rotation which is common in real world images.also its computationaly simple and faster making it practical.


# This function is responsible for spliting the training data into training and validation sets
def split_train_validation(df, validation_size=0.25):
    # Shuffle the dataframe first because we want randomness
    df = df.sample(frac=1).reset_index(drop=True)

    # Group the images by label because we want to keep the distribution consistent
    grouped = df.groupby('label')

    train_df = pd.DataFrame()
    validation_df = pd.DataFrame()

    # Going through each group which represents each class of images
    for _, group in grouped:
        n_validation = int(len(group) * validation_size)  # Calculate how many should go into validation
        validation_group = group[:n_validation]  # This slices the first part for validation
        train_group = group[n_validation:]  # The rest stays in training

        # Add these to our training and validation dataframes
        validation_df = pd.concat([validation_df, validation_group], axis=0)
        train_df = pd.concat([train_df, train_group], axis=0)

    # A good measure to shuffle them again after the split
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    validation_df = validation_df.sample(frac=1).reset_index(drop=True)

    return train_df, validation_df


# Function responsible for training with different K values to see which ones the best
def training(train_features, train_labels, validation_features, validation_labels):
    k_values = [1, 3, 5, 7]  # These are the Ks we are going to try out
    accuracies = []  # We wll keep track of accuracy for each K

    # Loop over each K value to test out our classifier
    for k in k_values:
        # im using the Manhattan distance here could be Euclidean but i just used manhattan
        knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

        # Training - we fit the model with our training data
        knn.fit(train_features, train_labels)

        # Now we predict on the validation set to see how well we did
        validation_preds = knn.predict(validation_features)

        # calculate the accuracy for these predictions
        accuracy = accuracy_score(validation_labels, validation_preds)
        accuracies.append(accuracy)  # Save this accuracy for this k value

    # Plotting the accuracies to visually compare which K did the best
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('KNN Validation Accuracy for Different K Values')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()

    # Finding the K that gave the highest accuracy
    best_k = k_values[np.argmax(accuracies)]
    print(f"The best K value based on validation accuracy is: {best_k}")

    return best_k


# Function to test how well our model does on the test set with the best K
def testing(best_k, train_features, train_labels, test_features, test_labels):
    # Setup the KNN using the best K we found from before and still using Manhattan
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')

    # Train again but this time with all the training data we have
    knn.fit(train_features, train_labels)

    #now we see this perform on the test data
    test_preds = knn.predict(test_features)

    # get the accuracy for these predictions
    accuracy = accuracy_score(test_labels, test_preds)
    print(f"Testing Accuracy with K={best_k}: {accuracy:.4f}")

    return accuracy


# image classification system
def main():
    # First we load the data.
    # used relative path. the main file is inside the databse folder as you said in the assignment.
    train_df = load_dataset('./Database/TrainingSet')
    test_df = load_dataset('./Database/TestingSet')

    # Split the training data to get some for validation, using a 75-25 split
    train_df, validation_df = split_train_validation(train_df)

    # We need to label encode the labels because our model needs numbers not string labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['label'].values)
    validation_labels = label_encoder.transform(validation_df['label'].values)
    test_labels = label_encoder.transform(test_df['label'].values)

    # Extract features using histogram-based method
    train_features_hist = hist_features(train_df['image_path'])
    validation_features_hist = hist_features(validation_df['image_path'])
    test_features_hist = hist_features(test_df['image_path'])

    # Extract features using LBP method
    train_features_lbp = mystery_features(train_df['image_path'])
    validation_features_lbp = mystery_features(validation_df['image_path'])
    test_features_lbp = mystery_features(test_df['image_path'])

    # Train and evaluate KNN classifiers using histogram based features
    best_k_hist = training(train_features_hist, train_labels, validation_features_hist, validation_labels)
    test_accuracy_hist = testing(best_k_hist, train_features_hist, train_labels, test_features_hist, test_labels)

    # Train and evaluate KNN classifiers using LBP features
    best_k_lbp = training(train_features_lbp, train_labels, validation_features_lbp, validation_labels)
    test_accuracy_lbp = testing(best_k_lbp, train_features_lbp, train_labels, test_features_lbp, test_labels)

    # Print out the final test accuracies to compare
    print(f"Accuracy using histogram-based features: {test_accuracy_hist:.2%}")
    print(f"Accuracy using LBP features: {test_accuracy_lbp:.2%}")


if __name__ == "__main__":
    main()
