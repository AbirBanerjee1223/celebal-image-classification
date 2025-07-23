import streamlit as st
import joblib
from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage import color
import cv2

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="LGBM Image Classifier",
    page_icon="⚡",
    layout="wide"
)

# ==============================================================================
# LOAD ARTIFACTS
# ==============================================================================
# This should be the path to your LGBM artifact file
ARTIFACT_PATH = 'image_classifier_artifacts_v1.joblib'

@st.cache_resource
def load_artifacts(path):
    """Loads all necessary components from the artifact file."""
    try:
        artifacts = joblib.load(path)
        return artifacts
    except FileNotFoundError:
        st.error(f"Artifact file not found at '{path}'. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the artifact file: {e}")
        return None

artifacts = load_artifacts(ARTIFACT_PATH)
if artifacts:
    model = artifacts.get('model')
    scaler = artifacts.get('scaler')
    label_encoder = artifacts.get('label_encoder')
    config = artifacts.get('config')
    # Validate that all required components are present
    if not all([model, scaler, label_encoder, config]):
        st.error("Artifact file is invalid or missing required components.")
        artifacts = None

# ==============================================================================
# FEATURE EXTRACTION PIPELINE (HOG-Only as per the config)
# ==============================================================================
def extract_hog_features(image, config):
    """
    Preprocesses an image and extracts HOG features.
    This function is tailored to the provided joblib file, which does not use color features.
    """
    try:
        # 1. Resize the image
        image_resized = cv2.resize(np.array(image), config['image_size'])
        
        # 2. Convert to grayscale for HOG
        gray_image = color.rgb2gray(image_resized)
        
        # 3. Extract HOG features
        hog_features = hog(gray_image,
                           orientations=config['features']['hog_orientations'],
                           pixels_per_cell=config['features']['hog_pixels_per_cell'],
                           cells_per_block=config['features']['hog_cells_per_block'],
                           transform_sqrt=True,
                           block_norm='L2-Hys',
                           visualize=False)
        
        return hog_features.reshape(1, -1)
    except Exception as e:
        st.error(f"Could not extract HOG features. Error: {e}")
        return None

# ==============================================================================
# UI SIDEBAR
# ==============================================================================
st.sidebar.title("Classifier Controls")

if artifacts:
    # Confidence Threshold Slider
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, value=0.40, step=0.05,
        help="Flag predictions with confidence below this value."
    )
    st.sidebar.markdown("---")

    # Model Card
    st.sidebar.header("Model Card")
    st.sidebar.info(
        """
        - **Model Type:** LightGBM (LGBM)
        - **Features:** HOG (Histogram of Oriented Gradients)
        - **Classes:** 6
        """
    )
    class_names = list(label_encoder.classes_)
    with st.sidebar.expander("Show All 6 Identifiable Classes"):
        st.write(class_names)
else:
    st.sidebar.error("Model artifacts not loaded.")

# ==============================================================================
# MAIN PAGE INTERFACE
# ==============================================================================
st.title("LightGBM Image Classifier")
st.write("Upload an image to classify it into one of the 6 trained categories using a HOG-based model.")

if not artifacts:
    st.warning("Application is not operational because the model artifacts could not be loaded.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Main layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            with st.spinner('Extracting HOG features and classifying...'):
                # Run the HOG feature extraction pipeline
                features = extract_hog_features(image, config)
                
                if features is not None:
                    # Scale features and predict
                    scaled_features = scaler.transform(features)
                    prediction_probs = model.predict_proba(scaled_features)
                    confidence_score = np.max(prediction_probs)
                    prediction_index = np.argmax(prediction_probs)
                    predicted_class_name = label_encoder.inverse_transform([prediction_index])[0]

                    st.success("Analysis Complete!")

                    # Display prediction with confidence check
                    if confidence_score >= confidence_threshold:
                        st.metric(label="Predicted Class", value=predicted_class_name.capitalize())
                        st.metric(label="Confidence", value=f"{confidence_score:.2%}")
                    else:
                        st.warning(f"Prediction Uncertain")
                        st.metric(label="Predicted Class", value=predicted_class_name.capitalize())
                        st.metric(label="Confidence", value=f"{confidence_score:.2%}",
                                  delta=f"Below {confidence_threshold:.0%} threshold", delta_color="inverse")
                    
                    st.markdown("---")

                    # Feature Details Expander
                    with st.expander("Feature Details"):
                        st.info(f"The image was converted into a HOG feature vector of size **{features.shape[1]}**.")