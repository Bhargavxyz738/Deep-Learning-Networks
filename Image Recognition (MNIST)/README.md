# Advanced MNIST: Variable-Length Sequence Recognition in PyTorch

This project demonstrates how to build, train, and optimize a Convolutional Neural Network (CNN) in PyTorch to solve a problem more complex than standard digit classification. The goal is to accurately recognize sequences of handwritten digits of varying lengths (1, 2, or 3 digits), which are dynamically stitched together from the original MNIST dataset. Through a systematic process of model improvement, data augmentation, and intelligent training, the final model achieves an exact sequence match accuracy of **99.25%** (Trained on 20 epochs) on the test set.

## Deep Learning Network Architecture

The network is a custom-designed Convolutional Neural Network (CNN) built to handle a single, wide image and produce multiple, parallel predictions. It follows a classic feature extraction and classification paradigm, adapted for this multi-task problem.

*   **Network Type:** Multi-Head Convolutional Neural Network.
*   **Input Layer:** The network accepts a fixed-size, single-channel grayscale image tensor of shape `[batch_size, 1, 28, 84]`. The height of 28 corresponds to the standard MNIST digit height, while the width of 84 accommodates the maximum possible sequence of three 28-pixel-wide digits.
*   **Output Layers:** The model has **three separate output layers** (or "heads"), one for each potential digit in the sequence. Each head is a fully-connected `Linear` layer that outputs 11 logits, corresponding to the 10 digits (0-9) and a special `blank_token` (10) used for padding.

### Architectural Breakdown:

1.  **Convolutional Backbone (Feature Extractor):** The core of the model consists of a deep stack of four `Conv2d` layers. This backbone scans the entire `28x84` input image to learn a hierarchy of spatial features—from simple edges and curves in the initial layers to more complex digit-part representations in the deeper layers. `ReLU` activations are used to introduce non-linearity, and `MaxPool2d` layers are interspersed to downsample the feature maps, making the learned features more robust to their exact location.
    
2.  **Fully-Connected Neck (Shared Representation):** After feature extraction, the final 2D feature map is flattened into a 1D vector. This vector is then passed through a large, fully-connected `Linear` layer to produce a 512-dimensional embedding. This dense vector serves as a high-level, compressed summary of the entire input image.
    
3.  **Regularization:** A `Dropout` layer (`p=0.5`) is applied to this shared representation during training. This is a critical regularization technique that randomly deactivates neurons, preventing the network from overfitting and forcing it to learn more robust, generalized features.
    
4.  **Prediction Heads:** The 512-dimensional vector is then fed in parallel to each of the three output heads. Each head independently interprets this shared summary to make its prediction for its assigned digit position.

## Advanced Training Techniques

Achieving high accuracy required moving beyond a basic training loop. The final model was trained using a combination of techniques designed to improve generalization, prevent overfitting, and ensure the model reached its true optimal performance.

*   **Data Augmentation:** To make the model robust to variations in handwriting, the training data was augmented on-the-fly. Each MNIST digit was subjected to random, minor transformations—including **rotation** (up to 10 degrees), **translation** (shifting horizontally and vertically by up to 10% of the image size), and **scaling** (zooming in or out by up to 10%)—before being stitched into a sequence. This forced the model to learn the essential, invariant features of each digit rather than memorizing their specific pixel patterns.

*   **Validation-Based Early Stopping:** The model was evaluated against a separate validation set (a 10% split of the training data) at the end of each epoch. Training was automatically configured to stop if the validation loss failed to improve for a set number of consecutive epochs (a "patience" of 5). This crucial step prevents overfitting by ensuring the model is saved at the point of peak generalization performance, rather than simply continuing to memorize the training data.

*   **Adaptive Learning Rate with `ReduceLROnPlateau`:** A fixed learning rate can be inefficient. This project employed an adaptive learning rate scheduler that monitored the validation loss. If the loss plateaued (failed to improve for 2 epochs), the learning rate was automatically reduced by a factor of 10. This allowed the model to make large, confident updates in the early stages of training and then switch to smaller, more precise updates to fine-tune its performance and find a better final solution, which proved critical for breaking through performance plateaus.

*   **Combined Loss Function:** Since the model has three output heads, the total loss for each batch was calculated by summing the individual Cross-Entropy losses from each head. This multi-task loss was then backpropagated through the entire network, allowing the shared convolutional backbone to learn features that were useful for all three prediction tasks simultaneously.

## Results and Performance

The final, improved model demonstrates exceptional performance on the task of variable-length sequence recognition. When evaluated on the unseen test set, the model achieves an **Exact Sequence Match Accuracy of 99.25%**.

This high accuracy means that the model correctly identified the entire sequence of digits—including the correct length and the value of each digit—in over 99 out of 100 cases. This is a strong indicator of the model's ability to generalize, which can be attributed to the robust training regimen involving extensive data augmentation and regularization.

### Prediction Visualization

The following image shows a random sample of 10 predictions from the test set. The model's predicted sequence is shown above the actual ground-truth label. The title is colored green for a correct match and red for an incorrect one. As shown, the model performs reliably on sequences of all possible lengths (1, 2, and 3 digits).

![Model Prediction Examples](https://i.postimg.cc/KcP7Y2Jb/download.png)

## How to Run

### Requirements
To run this project, you will need Python 3 and the following libraries:
- PyTorch
- TorchVision
- NumPy
- Matplotlib

You can install them using pip:
```bash
pip install torch torchvision numpy matplotlib
```

### Execution
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run the main script from your terminal:
```bash
python train.py
```
The script will automatically download the MNIST dataset, build the model, and begin the training process. You will see live progress in your terminal, including epoch-by-epoch loss, validation results, and notifications when the learning rate is adjusted or the best model is saved.

After training is complete, the script will load the best-performing model, evaluate it on the test set to print the final accuracy, and display a visualization of sample predictions.

### :warning: Note on Computational Requirements
This is a deep learning model that is computationally expensive to train.
*   **GPU Recommended:** Training is **highly recommended** on a machine with a CUDA-enabled GPU. The script will automatically use the GPU if it is available.
*   **Time Consuming:** Training on a GPU can take a significant amount of time (e.g.,  20 to 30mins , depending on the hardware). Training on a CPU will be exceptionally slow and may take several hours to complete(e.g., 4hr to 6hr).

## Conclusion

This project successfully demonstrates that by combining a well-designed multi-head CNN architecture with a suite of advanced training techniques, it is possible to solve complex computer vision tasks like variable-length sequence recognition with extremely high accuracy. The final 99.25% accuracy is a direct result of a systematic engineering approach, highlighting that modern deep learning success relies not just on the model architecture, but equally on intelligent data handling, robust regularization, and adaptive training procedures.
