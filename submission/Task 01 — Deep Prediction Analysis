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
The model produced a predicted label for each of the three selected test samples. The predicted label was obtained by selecting the output class with the highest probability.

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

During the forward pass, each test sample moves through the model one layer at a time. The input pixels are first transformed using the weights and biases learned during training. Each hidden layer creates a higher-level representation of the image.

The ReLU activation function is used in the hidden layers. It keeps positive activation values and changes negative values to zero:

ReLU(x) = max(0, x)

This introduces non-linearity into the model and allows it to learn complex patterns instead of only simple linear relationships. For image data, earlier neurons may respond to basic patterns such as edges and lines, while later neurons may combine these patterns into shapes that help identify the correct class.

The final layer uses the Softmax activation function. Softmax converts the output scores into a probability distribution across all possible classes. The class with the highest probability becomes the predicted label.

For a correct prediction, the forward pass extracted features that were strongly associated with the true class. As a result, the Softmax output assigned its highest probability to the correct label.

For an incorrect prediction, the sample may contain unclear, unusual, or overlapping features. The extracted representation may therefore be more similar to another class, causing Softmax to assign the highest probability to the wrong label. A high-confidence incorrect prediction means that the model strongly learned a misleading feature pattern.

During training, the Adam optimizer updated the model’s weights using gradients calculated through backpropagation. Adam adapts the update size for each parameter using estimates of the gradients’ average direction and magnitude. This helped the model reduce its loss and learn useful feature patterns efficiently.

However, Adam does not guarantee that every sample will be classified correctly. Its weight updates are based on the overall training data, so uncommon or ambiguous test samples may still be misclassified.

What Happened During the Forward Pass?

During the forward pass, the input image moved through the model layer by layer:

```text
28 × 28 image
      ↓
Flatten layer
      ↓
784 pixel values
      ↓
Dense layer with 64 neurons
      ↓
ReLU activation
      ↓
Output layer with 10 neurons
      ↓
Softmax activation
      ↓
Predicted label
```

First, the `Flatten` layer converted the image from a two-dimensional `28 × 28` matrix into a one-dimensional vector containing `784` pixel values. This layer does not learn patterns; it only changes the shape of the input.

Next, the `Dense` layer used the learned weights and biases to transform the pixel values into internal features that help distinguish between different digits.

The `ReLU` activation function introduced non-linearity by keeping positive values and converting negative values to zero:

```text
ReLU(x) = max(0, x)
```

This allowed the model to learn more complex patterns in the image, such as lines, curves, and combinations of pixels.

Finally, the output layer produced ten scores, one for each digit from `0` to `9`. The `Softmax` activation converted these scores into probabilities. The digit with the highest probability was selected as the predicted label.


## 5. Short Analysis

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
