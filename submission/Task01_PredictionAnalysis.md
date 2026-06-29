# Task 01 — Deep Prediction Analysis

## 1. Objective

The objective of this task is to analyze the predictions of a trained neural network on three samples from the test dataset. The analysis explains how the forward pass, activation functions, and the Adam optimizer contributed to each prediction.

## 2. Code Used

```python
import numpy as np
import matplotlib.pyplot as plt

sample_indices = [0, 1, 2]

for index in sample_indices:
    sample = x_test[index]
    true_label = y_test[index]

    # Add the batch dimension required by the model
    sample_input = sample.reshape(1, 28, 28)

    # Predict the probabilities of all classes
    probabilities = model.predict(sample_input, verbose=0)

    predicted_label = np.argmax(probabilities)
    confidence = np.max(probabilities)

    plt.imshow(sample, cmap="gray")
    plt.title(
        f"Predicted: {predicted_label} | "
        f"True: {true_label} | "
        f"Confidence: {confidence:.2%}"
    )
    plt.axis("off")
    plt.show()

    print("Predicted Label:", predicted_label)
    print("True Label:", true_label)
    print("Confidence:", confidence)
    print("Correct Prediction:", predicted_label == true_label)
    print("-" * 40)
```

## 3. Results

### Sample 1

* **Predicted Label:** `6`
* **True Label:** `6`
* **Confidence:** `0.9998164`
* **Prediction Status:** `Correct`

### Sample 2

* **Predicted Label:** `1`
* **True Label:** `1`
* **Confidence:** `0.99873286`
* **Prediction Status:** `Correct`

### Sample 3

* **Predicted Label:** `2`
* **True Label:** `2`
* **Confidence:** `0.99996984`
* **Prediction Status:** `Correct`

## 4. Short Analysis

### Sample 1 Analysis

During the forward pass, the image pixels were passed through the layers of the neural network. Each layer applied learned weights and biases to transform the input into increasingly useful representations.

The ReLU activation function in the hidden layer kept positive values and converted negative values to zero. This introduced non-linearity and helped the model detect important patterns in the image.

The Softmax activation function in the output layer converted the final values into probabilities for the possible classes. The class with the highest probability was selected as the predicted label.

The Adam optimizer contributed during training by updating the model weights based on the gradients and reducing the loss. These learned weights caused the model to assign the highest probability to class `[predicted label]`.

The prediction was `correct` because `the detected image features matched the correct class`.

### Sample 2 Analysis

During the forward pass, the sample was transformed through the network using the learned weights and biases. ReLU activated useful positive features in the hidden layer, while Softmax produced the final class probabilities.

Adam adjusted the weights during training so that patterns similar to this sample would produce a higher probability for the appropriate class.

The prediction was `[correct/incorrect]` because `[explain the likely reason based on the image shape, clarity, or similarity to another class]`.

### Sample 3 Analysis

The model processed the image through a forward pass. The hidden layers used ReLU to preserve useful feature responses, and the output layer used Softmax to calculate the probability of each class.

Adam helped the model learn suitable weights by adapting the weight updates during training.

The prediction was `[correct/incorrect]` because `[explain why the model selected this class]`.

## 5. Key Takeaway

The model makes predictions by transforming image pixels through learned layers, using ReLU to detect useful non-linear features and Softmax to select the most probable class. The quality of the prediction depends on how well the weights learned by Adam represent the patterns found in the test image.
