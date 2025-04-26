# Cat-Breed-Image-Classifier

---

##  Dataset

The dataset was compiled from multiple Kaggle sources and structured into a `train/`, `val/`, and `test/` split.  
It includes the following breeds:

- Bombay
- British Shorthair
- Calico
- Domestic Long Hair
- Exotic Shorthair
- Himalayan
- Maine Coon
- Russian Blue
- Scottish Fold
- Siamese
- Tortoiseshell
- Rex
- Sphynx

Some breeds were underrepresented (e.g., Sphynx, Rex), leading to a class imbalance, which was addressed in model training.

---

## Methods and Techniques

- **Data Augmentation**: Random flips, rotations, zooms, and contrast changes applied during training.
- **Class Weights**: Computed manually based on true dataset distribution to counteract class imbalance during loss computation.
- **Model Architecture**:
  - **Transfer Learning** with ResNet50 backbone pretrained on ImageNet
  - Custom fully-connected layers for final classification
  - Initial freezing of convolutional layers, followed by fine-tuning
- **Callbacks**:
  - Early stopping to prevent overfitting
- **Optimizers and Loss**:
  - Adam optimizer
  - Sparse categorical crossentropy loss
- **Visualization Tools**:
  - Class distribution stacked bar charts
  - Random sample image grids
  - Augmentation previews
  - t-SNE feature space projections
  - Grad-CAM heatmaps to interpret model focus

---

##  Results

- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-Score per breed
- **Visualizations**:
  - Confusion Matrix
  - t-SNE plots showing feature clustering
  - Grad-CAM overlays highlighting areas the model used for prediction

---

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/ktmcquinn/cat-breed-classification.git
    cd Cat-Breed-Image-Classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the notebook:
    ```bash
    jupyter notebook notebooks/Cat_Breed_Classifier.ipynb
    ```

4. (Optional) Update dataset paths in the notebook to match your environment.

---

##  Lessons Learned

- Class imbalance can significantly bias model learning without correction.
- Fine-tuning a pretrained network (Transfer Learning) drastically improves classification performance.
- Visualization tools like Grad-CAM and t-SNE are critical for understanding model behavior beyond raw metrics.

---

##  Future Work

- Explore lightweight models (e.g., MobileNetV2) for faster inference.
- Implement ensemble models to boost accuracy further.
- Investigate using larger datasets or synthetic data generation for underrepresented breeds.

---

##  Acknowledgements

- Kaggle datasets contributors for cat breed images
- TensorFlow/Keras team for deep learning frameworks
- scikit-learn for evaluation metrics
