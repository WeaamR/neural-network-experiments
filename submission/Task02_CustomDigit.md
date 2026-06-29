## Task 02 — Custom Image Generalization Test

### 1. Objective

Draw a custom handwritten digit, preprocess it to match the model's expected input format, run a prediction, and analyze whether the model generalizes beyond the training distribution — referencing representation learning, distribution shift, and augmentation.

---

### 2. Code Used

```python
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
# Create folders if they do not already exist
os.makedirs("data/custom_digits", exist_ok=True)
os.makedirs("results/predictions", exist_ok=True)

# Upload the custom handwritten digit
uploaded = files.upload()

# Get the uploaded filename
image_filename = next(iter(uploaded))
print("Uploaded image:", image_filename)

# Save a copy of the original image inside the project
original_image_path = "data/custom_digits/task02_original_digit.png"
shutil.copy(image_filename, original_image_path)

# Open the original image
img = Image.open(image_filename)

# Convert the image to grayscale
gray_image = img.convert("L")

# Resize the image to match MNIST
resized_image = gray_image.resize(
    (28, 28),
    Image.Resampling.LANCZOS
)

# Convert the image into a NumPy array
custom_image = np.array(resized_image, dtype=np.float32)

# MNIST uses a black background and a white digit.
# Invert the image when the background is mostly white.
if custom_image.mean() > 127:
    custom_image = 255.0 - custom_image

# Normalize pixel values from [0, 255] to [0, 1]
custom_image = custom_image / 255.0

# Display the final preprocessed image
plt.imshow(custom_image, cmap="gray")
plt.title("Preprocessed Custom Digit")
plt.axis("off")
plt.show()

# Save the exact preprocessed image used by the model
preprocessed_image_path = (
    "results/predictions/task02_preprocessed_digit.png"
)

plt.imsave(
    preprocessed_image_path,
    custom_image,
    cmap="gray",
    vmin=0,
    vmax=1
)
# Add the batch dimension:
# (28, 28) → (1, 28, 28)
custom_image_input = custom_image.reshape(1, 28, 28)

# Run prediction
prediction = model.predict(custom_image_input, verbose=0)

# Get the class with the highest probability
predicted_label = np.argmax(prediction[0])

# Get the confidence of the predicted class
confidence = prediction[0][predicted_label]

print("Predicted label:", predicted_label)
print(f"Confidence: {confidence:.4f}")

result_path = "results/predictions/task02_prediction.txt"

with open(result_path, "w") as file:
    file.write("Task 2 — Custom Image Generalization Test\n")
    file.write(f"Original image: {original_image_path}\n")
    file.write(
        f"Preprocessed image: {preprocessed_image_path}\n"
    )
    file.write(f"Predicted label: {predicted_label}\n")
    file.write(f"Confidence: {confidence:.4f}\n")

print("Prediction result saved to:", result_path)
```

---

### 3. Results

```
Digit Drawn: 4
Predicted Label: 4
Prediction Confidence: 0.8963
Prediction Status: Correct
```
```
Digit Drawn: 8
Predicted Label: 8
Prediction Confidence: 0.5193
Prediction Status: Correct
```
```
Digit Drawn: 3
Predicted Label: 3
Prediction Confidence: 1.0000
Prediction Status: Correct
```
The custom image was converted to grayscale, resized to 28 × 28, normalized to the range [0, 1], and reshaped to (1, 28, 28) before being passed to the model.

The final preprocessed image used a white digit on a black background to make it more similar to the MNIST images used during training.

---

### 4. Short Analysis

All three custom digits were classified correctly, suggesting that the model can generalize beyond the MNIST test set to new hand-drawn inputs. However, the differences in confidence show that some custom digits matched the learned MNIST representations more closely than others.

#### Digit 4 — Confidence: 89.63%

The digit `4` was correctly classified with high confidence. The original image had a black digit on a white background, which is the opposite of the MNIST format. Therefore, the color-inversion step was necessary to produce a white digit on a black background before prediction.

The digit also uses a printed style with sharp angles and straight lines. Although this style may differ slightly from typical handwritten MNIST samples, the model still recognized the main structural features of `4`. This indicates that the preprocessing pipeline successfully matched the expected input format and that the learned representation was sufficiently robust to handle moderate style variation.

#### Digit 8 — Confidence: 51.93%

The model correctly classified the digit `8`, but with considerably lower confidence. Its two-loop structure supported the correct prediction, while the thick stroke and uneven loop sizes made it less similar to typical MNIST samples.

These variations may have caused the image to share features with similar classes such as `0`, `3`, or `6`. As a result, the Softmax probability was likely distributed across several classes instead of being concentrated strongly on class `8`. This suggests a small distribution shift between the custom image and the training data.

#### Digit 3 — Confidence: 100.00%

The digit `3` was correctly classified with very high confidence. It was clear, centered, and contained two visible curved sections that closely matched the features associated with class `3`.

The displayed confidence of `100.00%` indicates that the model assigned nearly all the probability to class `3`. However, this value may have been rounded when printed and should not be interpreted as a guarantee that every similar image would be classified correctly.

### Did the Model Classify the Digits Correctly?

Yes. The model correctly classified all three custom digits: `4`, `8`, and `3`. The predictions for `4` and `3` had high confidence, while the lower confidence for `8` showed that the model found its features more ambiguous.

### Relationship to Representation Learning

These results suggest that the neural network learned internal representations of digit shapes instead of merely memorizing individual training images. The hidden layer transformed raw pixel values into useful patterns related to lines, curves, intersections, and loop structures.

When a custom image contained features similar to those learned from MNIST, the model assigned the correct class the highest probability. The lower confidence for digit `8` demonstrates that these representations are affected by differences in handwriting style, stroke thickness, position, and shape.

### Effect of the Lack of Data Augmentation

The model was trained without data augmentation, so it had limited exposure to variations such as thicker strokes, uneven shapes, rotations, shifts, and different digit sizes.

This may explain the lower confidence for digit `8`. Training with augmented images could help the model learn more flexible representations and improve its confidence and generalization when processing unfamiliar handwriting styles.

---

### 5. Key Takeaway

The model generalized successfully to the three custom digits, but the confidence differences show that prediction quality depends on how closely each new image matches the visual patterns and representations learned from the training data.
