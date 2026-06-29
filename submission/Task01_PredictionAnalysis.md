# Task 01 — Deep Prediction Analysis

## 1. Objective

The objective of this task is to analyze the predictions of a trained neural network on three samples from the test dataset. The analysis explains how the forward pass, activation functions, and the Adam optimizer contributed to each prediction.

## 2. Code Used

```python

def analyze_sample(sample_index):
    """
    Predict and display one sample from the test set.

    Parameters:
        sample_index (int): The index of the sample in x_test and y_test.
    """

    # Select the sample and its true label
    sample_x = x_test[sample_index]
    sample_y = y_test[sample_index]

    print("Sample index:", sample_index)
    print("Actual value:", sample_y)

    # Add a batch dimension: (28, 28) -> (1, 28, 28)
    sample_batch = sample_x.reshape(1, 28, 28)

    # Make the prediction
    prediction = model.predict(sample_batch, verbose=0)

    # Select the class with the highest probability
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)

    print("Predicted value:", predicted_label)
    print("Confidence:", confidence)

    # Check whether the prediction is correct
    if predicted_label == sample_y:
        print("Result: Correct prediction")
    else:
        print("Result: Incorrect prediction")

    # Display the image
    plt.imshow(sample_x, cmap="gray")

    plt.axis("off")
    plt.show()

# Call the function by passing the sample index:

# Sample 1
analyze_sample(666)

# Sample 2
analyze_sample(777)

# Sample 3
analyze_sample(888)

```

## 3. Results
The model produced a predicted label for each of the three selected test samples. The predicted label was obtained by selecting the output class with the highest probability.

### Sample 1

* **Predicted Label:** `7`
* **True Label:** `7`
* **Confidence:** `0.99839216`
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

### How the forward pass works:
Each input sample passes through the network layer by layer. In a typical Dense network, every neuron computes a weighted sum of its inputs plus a bias: `z = W·x + b`. This raw value is then passed to an activation function.

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

### Role of ReLU:
Hidden layers use ReLU (`f(z) = max(0, z)`), which introduces non-linearity while keeping gradients clean for positive values. ReLU allows the network to learn complex patterns without the vanishing gradient problem common with sigmoid or tanh.

### Role of Softmax:
The final layer uses Softmax, which converts raw logits into a probability distribution across all classes. The class with the highest probability becomes the predicted label. This is why we use `np.argmax()` on the output — we're selecting the most confident class.

### Adam optimizer's influence:
Adam adapts the learning rate for each weight individually using estimates of first and second moments of the gradients. During training, Adam helped the model converge faster and more stably than standard SGD. The high confidence scores (>0.97) indicate that the weights were well-optimized — Adam guided the model toward a sharp, confident decision boundary for these samples.

### Why predictions were correct:
The model likely saw many similar patterns during training. The features that activate strongly in earlier layers (edges, strokes, curves) were reliably mapped to the correct class through the learned weights. The high confidence scores suggest that Adam's adaptive updates pushed the weights toward a sharp, well-defined decision boundary for these sample classes.

5. Key Takeaway
The combination of ReLU activations for deep feature extraction and Softmax for probabilistic output, trained with Adam's adaptive updates, enabled the model to make confident and accurate predictions — showing that a well-trained network learns robust internal representations, not just memorized patterns.


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
