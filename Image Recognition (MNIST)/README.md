# Advanced MNIST: Variable-Length Sequence Recognition in PyTorch

This project documents an end-to-end deep learning workflow in PyTorch, chronicling the evolution from a high-accuracy but brittle baseline model to a highly robust, generalized Convolutional Neural Network (CNN). The primary goal is to solve a complex computer vision challenge that goes far beyond simple classification: **accurately recognizing sequences of handwritten digits of varying lengths (1, 2, or 3)**.

The journey highlights a critical concept in machine learning: the trade-off between specialization and robustness. While an initial model achieved a staggering **99.25%** accuracy on a clean test set, it was a "hollow victory," as the model failed completely when faced with minor visual distortions. The final, improved model, trained for **28 epochs (approx. 47 minutes on a GPU)**, is the true success story. It achieves a slightly lower clean-data accuracy but demonstrates outstanding, reliable performance across a battery of real-world challenges like noise, rotation, and color inversion.

## The Core Challenge: From Classification to Sequence Recognition

Standard MNIST classification involves mapping a single `28x28` image to one of 10 labels. Our problem is significantly more complex:

1.  **Variable-Length Input:** The model must correctly identify sequences of one, two, or three digits.
2.  **Spatial Awareness:** It must understand *where* one digit ends and another begins within a single, wide image.
3.  **Multi-Task Prediction:** Instead of one output, it must produce multiple, position-dependent outputs.
4.  **End-of-Sequence Detection:** The model needs a mechanism to determine the actual length of the sequence and ignore any padded, empty space.

To address this, the entire pipeline, from data generation to model architecture, was custom-engineered.

## The Custom Dataset: `MultiDigitVariableMNIST`

Standard datasets are insufficient for this task. Therefore, we created a custom PyTorch `Dataset` class that dynamically generates variable-length digit sequences on-the-fly from the base MNIST dataset. This process is the foundation of the entire project.

Here is the step-by-step generation process for a single sample:

1.  **Determine Sequence Length:** A random integer between 1 and 3 is chosen. This ensures the model sees a balanced mix of all possible sequence lengths during training.
2.  **Sample Individual Digits:** The required number of digits (e.g., two for a length-2 sequence) are randomly sampled from the base MNIST training set.
3.  **Apply Augmentations:** **Crucially, data augmentations (rotation, translation, etc.) are applied to each digit *individually***. This simulates the natural variation where each character in a sequence might be written slightly differently.
4.  **Concatenate into a Sequence:** The augmented digit tensors are concatenated horizontally (`torch.cat` along the width dimension) to form a single, wide image. For example, two `1x28x28` digits become one `1x28x56` image.
5.  **Pad to Maximum Length:** To enable batching, all images must have a uniform size. The generated sequence image is padded with a constant background value on the right side until it reaches the maximum possible width of `84` pixels (`28 * 3`).
6.  **Generate Padded Labels:** A corresponding label tensor is created. For a sequence like "7, 3", the label is `[7, 3, 10]`. The `10` is our special **`blank_token`**, which explicitly teaches the model to recognize the end of a sequence.

An input-output pair for the model looks like this:
*   **Input:** A single image tensor of shape `[1, 28, 84]`.
*   **Output:** A label tensor of shape `[3]`, e.g., `torch.tensor([digit_1, digit_2, blank_token])`.

## Architectural Deep Dive: A Multi-Head CNN for Sequence Recognition

To tackle this unique data format, we designed a custom multi-head CNN. This architecture is engineered to process the entire sequence image holistically and then make parallel, position-aware predictions.

#### Input Layer
The network is designed to accept a fixed-size, single-channel grayscale image tensor of shape `[batch_size, 1, 28, 84]`.

#### Convolutional Backbone (The "Eye" of the Network)
This is a shared feature extractor that processes the entire input image at once. It consists of a stack of four `Conv2d` layers.
*   **Hierarchical Feature Learning:** The initial layers learn low-level features like edges and curves. Deeper layers combine these to recognize more complex patterns like loops and entire digit parts.
*   **Non-Linearity (`ReLU`):** Rectified Linear Unit activations are applied after each convolution. This allows the network to learn complex, non-linear relationships between pixels.
*   **Spatial Invariance (`MaxPool2d`):** Pooling layers periodically downsample the feature maps. This makes the learned features more robust to their exact location, meaning the model can recognize a "7" whether it's slightly to the left or right.

#### Fully-Connected Neck (The "Brain" of the Network)
After the final pooling layer, the resulting 2D feature map is flattened into a long 1D vector. This vector is then passed through a large `Linear` layer to produce a **512-dimensional embedding**. This dense vector acts as a high-level, compressed summary of all the important features detected across the *entire* input image. It represents the network's holistic understanding of the sequence.

#### Regularization (The "Safety Net")
A `Dropout` layer (`p=0.5`) is applied to this 512-dimensional vector during training. It randomly deactivates 50% of the neurons for each training pass. This is a crucial technique that prevents the network from becoming over-reliant on any single feature and forces it to learn more robust, redundant representations, massively improving its ability to generalize.

#### Parallel Prediction Heads (The "Mouths" of the Network)
This is the architectural key to solving the sequence problem. The single 512-dimensional summary vector is fed **in parallel to three separate and independent `Linear` layers** (the "heads").
*   **Position-Specific Responsibility:** Each head is trained to be a specialist for its position (Head 1 predicts the first digit, Head 2 the second, etc.).
*   **The `blank_token`:** Each head outputs **11 logits** (scores) corresponding to digits 0-9 and our special `blank_token`. This is what allows the model to handle variable lengths. For an image of the digit "8", the model is trained to have Head 1 predict "8", Head 2 predict "blank", and Head 3 predict "blank".

## Advanced Training and Optimization Strategy

A sophisticated architecture demands a sophisticated training regimen. The final robust model was trained using a combination of powerful techniques:

*   **Combined Loss Function:** The total loss for a given sequence is the **sum of the individual `CrossEntropyLoss` from all three heads**. This multi-task loss is backpropagated through the entire network. This masterfully trains the shared convolutional backbone to learn features that are universally useful for identifying digits, regardless of their position in the sequence.

*   **Adaptive Learning Rate (`ReduceLROnPlateau`):** Instead of a fixed learning rate, we used a scheduler that monitors the validation loss. If the loss stagnates, the learning rate is automatically reduced. This allows the model to make rapid progress early on ("large steps") and then fine-tune its weights with precision later in training ("small steps") to find a better final solution.

*   **Validation-Based Early Stopping:** To prevent overfitting, the model was saved only when its performance on an unseen validation set improved. If the validation loss failed to improve for 5 consecutive epochs, training was automatically halted. This ensures the final model is captured at its moment of **peak generalization power**.

## Results & Analysis: The Specialist vs. The Generalist

The performance comparison starkly reveals the project's core lesson. The evaluation was performed using a separate, rigorous script (`evaluation.py`).

| Metric                                        | old_model.py (The Specialist)            | **new_model.py (The Robust Generalist)**     |
| --------------------------------------------- | ------------------------------------------ | ------------------------------------------- |
| **Training Focus**                            | Maximize accuracy on clean data            | **Maximize robustness & generalization**    |
| **Exact Sequence Accuracy (Clean Data)**      | **99.25%** (Excellent but brittle)         | 97.50% (Slightly lower but reliable)      |
| **Robustness (Accuracy on Inverted Colors)**  | **~0%** (Complete failure)                 | **97.38%** (Excellent performance)          |
| **Robustness (Accuracy on Heavy Rotation)**   | **0.3%** (Complete failure)                | **97.28%** (Excellent performance)          |
| **Robustness (Accuracy on Gaussian Noise)**   | **~0%** (Complete failure)                 | **97.22%** (Excellent performance)          |
| **Training Time (GPU)**                       | ~20 mins (20 Epochs)                       | **~47 mins (28 Epochs)**                    |

The 99.25% accuracy of the baseline model is a **mirage of performance**. It demonstrates mastery over a single, clean data distribution but is utterly useless in practice. The final model's slightly lower accuracy on clean data is a small price to pay for its **consistent, dependable, and robust performance** across all conditions, making it a true engineering success.

### Prediction Visualizations

#### Final, Robust Model (The Generalist)
The image below shows predictions from the **final, robust model**. The title is colored green for correct matches, demonstrating its reliability across sequences of all possible lengths on the standard (clean) test set.

![New Model Prediction Examples](https://i.postimg.cc/yNhGZX3d/541146839-1304191204586271-1447129507572982572-n.png)

#### Original, Baseline Model (The Specialist)
This visualization shows the baseline model's near-perfect performance on the clean test data. While impressive, this high accuracy did not translate to robustness, as shown in the table above.

![Old Model Prediction Examples](https://i.postimg.cc/KcP7Y2Jb/download.png)

## How to Run

### Requirements
- Python 3
- PyTorch & TorchVision
- NumPy
- Matplotlib

Install them via pip:
```bash
pip install torch torchvision numpy matplotlib
```

### Execution
1. Clone this repository to your local machine.
2. Navigate to the project directory.

#### To train the **Final, Robust Model**:
```bash
python new_model.py
```

#### To train the **Original, Baseline Model**:
```bash
python old_model.py
```
Each script is self-contained and will handle data downloading, model building, training, and final evaluation.

### :warning: Pre-trained Models

Please note that due to file size limitations, the pre-trained model weights (e.g., `best_model_improved.pth`) are **not included** in this GitHub repository. This repository provides all the source code required to reproduce the results and train the models from scratch. The final trained model file can be provided upon request.

### :warning: Note on Computational Requirements
*   **GPU Recommended:** Training is computationally expensive and is highly recommended on a CUDA-enabled GPU.
*   **Time Consuming:** Training the final model on a GPU takes approximately **45-60 minutes**. On a CPU, this process will be exceptionally slow and may take several hours.

## Conclusion

This project is a comprehensive case study in building robust deep learning systems. It demonstrates that solving complex problems requires moving beyond off-the-shelf solutions. By engineering a custom data pipeline, designing a bespoke multi-head architecture, and employing a suite of advanced training techniques, we developed a model capable of high performance. More importantly, by prioritizing generalization through aggressive data augmentation, we successfully navigated the trade-off between raw accuracy and real-world robustness, creating a system that is not only accurate but, crucially, **reliable**.
