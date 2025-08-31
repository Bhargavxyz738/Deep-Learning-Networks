# Advanced MNIST: Variable-Length Sequence Recognition in PyTorch

This project showcases an end-to-end deep learning workflow in PyTorch, evolving from a high-accuracy but brittle baseline to a highly robust, generalized Convolutional Neural Network (CNN). The primary goal is to solve a complex computer vision challenge: accurately recognizing sequences of handwritten digits of varying lengths (1, 2, or 3).

The journey highlights a critical concept in machine learning: the trade-off between specialization and robustness. While the initial model achieved a staggering **99.25%** accuracy on clean test data, it failed completely when faced with visual distortions. The final, improved model, trained for **28 epochs (approx. 47 minutes on a GPU)**, achieves slightly lower clean-data accuracy but demonstrates outstanding resilience to real-world challenges like noise, rotation, and color inversion.

## Model Architectures and Training Philosophies

The project contains the code and results for two models. While they share the same underlying `CNN_VariableDigit` architecture, they are products of two vastly different training philosophies, resulting in dramatically different performance characteristics.

### The Final Model (`new_model.py`): The Robust Generalist

This is the flagship model of the project, engineered for real-world reliability. It was trained with a heavy emphasis on data augmentation to ensure it could generalize well beyond the clean training set.

*   **Training Philosophy:** Maximize robustness and the ability to generalize to unseen variations.
*   **Key Technique:** Aggressive on-the-fly data augmentation (rotation, translation, scaling) and strong regularization (`Dropout`).

### The Original Model (`old_model.py`): The Brittle Specialist

This model represents the baseline approach. It was trained with lighter data augmentation, which allowed it to hyper-specialize on the patterns present in the clean MNIST dataset.

*   **Training Philosophy:** Maximize accuracy on the clean, in-distribution test set.
*   **Key Technique:** Lighter data augmentation (`RandomAffine` with smaller ranges).

#### Common Architecture (`CNN_VariableDigit`)

Both models use the same powerful multi-head architecture:
1.  **Deeper Convolutional Backbone:** A stack of four `Conv2d` layers acts as a powerful feature extractor.
2.  **Fully-Connected Neck:** A dense layer creates a high-level 512-dimensional summary of the entire input image.
3.  **Regularization:** A `Dropout` layer (`p=0.5`) is used to combat overfitting.
4.  **Parallel Prediction Heads:** Three independent output heads predict the digit for each potential position in the sequence.

## Results: The Specialist vs. The Generalist

The performance comparison starkly reveals the trade-offs. The evaluation was performed using a separate, rigorous script that tested the models against a variety of visual challenges.

| Metric                                        | Baseline Model (The Specialist)            | **Final Model (The Robust Generalist)**     |
| --------------------------------------------- | ------------------------------------------ | ------------------------------------------- |
| **Training Focus**                            | Maximize accuracy on clean data            | **Maximize robustness & generalization**    |
| **Exact Sequence Accuracy (Clean Data)**      | **99.25%** (Excellent but brittle)         | 97.50% (Slightly lower but reliable)      |
| **Robustness (Accuracy on Inverted Colors)**  | **~0%** (Complete failure)                 | **97.38%** (Excellent performance)          |
| **Robustness (Accuracy on Heavy Rotation)**   | **0.3%** (Complete failure)                | **97.28%** (Excellent performance)          |
| **Robustness (Accuracy on Gaussian Noise)**   | **~0%** (Complete failure)                 | **97.22%** (Excellent performance)          |
| **Training Time (GPU)**                       | ~20 mins (20 Epochs)                       | **~47 mins (28 Epochs)**                    |

The results are clear: the baseline model, while impressive on paper with its 99.25% accuracy, is practically useless in any scenario where the input data is not perfectly clean. The final model is a true engineering success, providing high-level, reliable performance across all tested conditions.

### Prediction Visualization

The image below shows predictions from the **final, robust model**. The title is colored green for correct matches, demonstrating its reliability across sequences of all possible lengths.

![New Model Prediction Examples](https://i.postimg.cc/yNhGZX3d/541146839-1304191204586271-1447129507572982572-n.png)

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

This project successfully demonstrates the critical importance of robust training practices in deep learning. While achieving a high score on a clean test set is a good first step, building a truly effective model requires a focus on generalization. By evolving from a brittle "specialist" model to a resilient "generalist" through the use of aggressive data augmentation and regularization, we developed a system that can solve a complex sequence recognition task with extremely high and, most importantly, **reliable** accuracy across a wide range of conditions.
