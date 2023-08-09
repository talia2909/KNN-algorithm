#KNN Algorithm

## Introduction

The `KNN Algorithm` script implements the K-Nearest Neighbors (KNN) algorithm to classify handwritten digits from the renowned MNIST database. Specifically, this script focuses on distinguishing between the digits 3 and 5.

## Features

1. **MNIST Data Loading**:
    - Automatically loads the MNIST dataset for digits 3 and 5 into separate training, validation, and test sets.

2. **Visualization**:
    - Provides a visual representation of a sample digit from the dataset, aiding in manual verification and understanding.

3. **KNN Classifier**:
    - Implements the KNN algorithm, determining the classification of a sample based on its \( k \) closest neighbors in the training set.
    - Employs the validation set to optimize the choice of \( k \) for best classification accuracy.

4. **Result Export**:
    - Saves the classification results of the test dataset to a text file, with the filename corresponding to the student's ID.

## Usage

1. **Dependencies**:
    - Ensure you have the required libraries installed: `json`, `numpy`, and the custom module `utils`.
    - Ensure the MNIST dataset file `MNIST_3_and_5.mat` is available in the working directory or adjust the path accordingly.

2. **Student ID**:
    - Update the `ID` variable at the beginning of the script with the relevant student or user ID.

3. **Execution**:
    - Run the script. This will load the data, display a sample image, apply the KNN algorithm, and save the classification results for the test samples.

4. **Output**:
    - After successful execution, a text file named `<Student_ID>.txt` will be generated, containing the classification results.

## Conclusion

The `KNN Algorithm` script offers a straightforward implementation of the KNN algorithm tailored for classifying specific digits from the MNIST database. It's designed primarily as a homework assignment for an Information Theory course but can be adapted or expanded for other classification tasks.
