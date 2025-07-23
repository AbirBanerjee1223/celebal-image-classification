# Image Classification with Traditional Machine Learning

This project demonstrates a complete, end-to-end pipeline for image classification using traditional computer vision techniques and machine learning models. The primary goal is to classify images from a custom dataset into one of several categories.

The project includes scripts for data preprocessing, feature extraction (HOG and Color Histograms), data augmentation to handle class imbalance, model training, and hyperparameter tuning. It culminates in two interactive web applications built with Streamlit for real-time image classification.

-----

## 🚀 Key Features

  * **Advanced Feature Extraction**: Combines **Histogram of Oriented Gradients (HOG)** to capture object shape with **Color Histograms** to capture color information, creating a robust feature vector for each image.
  * **Data Augmentation**: Automatically identifies and augments minority classes to create a more balanced dataset, significantly improving model performance and fairness.
  * **Multi-Model Training**: Trains and evaluates multiple models (**SVM**, **RandomForest**, and **LightGBM**) to find the best performer for the given task.
  * **Interactive Web Apps**: Comes with two Streamlit applications for real-time prediction, featuring:
      * An easy-to-use file uploader.
      * A confidence threshold slider to filter uncertain predictions.
      * A "Model Card" providing details about the active model.

-----

## 📂 Project Structure

```
.
├── caltech_10_classes/         # Source dataset with 10 classes
├── caltech_10_classes_augmented/ # Auto-generated balanced dataset
│
├── image_classifier_artifacts_v1.joblib  # Artifacts for the 15-class LGBM model
├── image_classifier_artifacts_v1_balanced.joblib # Artifacts for the final 10-class RF model
│
├── app.py                      # Main Streamlit app for the best model (RandomForest)
├── app_lgbm.py                 # Secondary Streamlit app for the LGBM model
├── model_setup.ipynb        # The complete script for augmentation and training
└── README.md                   # This file
```

-----

## 🛠️ Setup and Installation

To get this project running locally, follow these steps.

**1. Clone the repository (if applicable):**

```bash
git clone (https://github.com/AbirBanerjee1223/celebal-image-classification)
cd celebal-image-classification
```

**2. Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the required packages:**
The `requirements.txt` file contains all necessary libraries.

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, create one with the following content:

```txt
streamlit
scikit-learn
numpy
pandas
opencv-python-headless
scikit-image
lightgbm
joblib
```

-----

## ⚙️ Usage

Follow these steps to train a model and run the web application.

### Step 1: Prepare Your Data

Place your source images in the `caltech_10_classes` directory (or your own source folder). The data must be organized into sub-folders where each sub-folder's name corresponds to a class label.

### Step 2: Train the Model

The training process is fully automated. Simply run the `training_pipeline.py` script.

```bash
python training_pipeline.py
```

This script will:

1.  Create the `caltech_10_classes_augmented` directory.
2.  Augment the minority classes and create a balanced dataset inside it.
3.  Extract combined HOG and Color features.
4.  Tune, train, and evaluate the SVM, RandomForest, and LightGBM models.
5.  Save the best-performing model and all related components into a `.joblib` artifact file.

### Step 3: Run the Web Application

This project includes two applications.

  * **To run the main app (with the best RandomForest model):**
    ```bash
    streamlit run app.py
    ```
  * **To run the secondary app (with the LGBM model):**
    ```bash
    streamlit run app_lgbm.py
    ```

Navigate to the local URL provided in your terminal to interact with the classifier.

-----

## 💡 Future Improvements

  * **Deep Learning Integration**: Experiment with a simple Convolutional Neural Network (CNN) and compare its performance to the traditional ML models.
  * **Advanced Augmentation**: Incorporate more advanced augmentation techniques like random cropping, noise injection, and elastic distortions.
  * **Deployment**: Deploy the main Streamlit application to a cloud service like Streamlit Community Cloud or Heroku for public access.
  * **CI/CD Pipeline**: Implement a continuous integration and deployment pipeline using GitHub Actions to automate testing and deployment.
