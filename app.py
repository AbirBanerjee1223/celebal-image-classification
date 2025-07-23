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
    page_title="Pro Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# ==============================================================================
# LOAD ARTIFACTS
# ==============================================================================
# This should be the path to your NEWEST joblib file
ARTIFACT_PATH = 'image_classifier_artifacts_v4_balanced.joblib'

@st.cache_resource
def load_artifacts(path):
    """Loads all necessary components from the artifact file."""
    try:
        artifacts = joblib.load(path)
        return artifacts
    except FileNotFoundError:
        st.error(f"Artifact file not found at '{path}'. Please ensure it is in the same directory.")
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
    # Check if all components are loaded
    if not all([model, scaler, label_encoder, config]):
        st.error("Artifact file is invalid. It's missing one or more required components.")
        artifacts = None # Invalidate artifacts if incomplete

# ==============================================================================
# FEATURE EXTRACTION PIPELINE (Must match training)
# ==============================================================================
def extract_combined_features(image, config):
    """Preprocesses an image and extracts the combined HOG + Color feature vector."""
    try:
        # Resize using OpenCV for consistency
        image_resized = cv2.resize(np.array(image), config['image_size'])
        
        # --- HOG Features ---
        gray_image = color.rgb2gray(image_resized)
        hog_feats = hog(gray_image,
                       orientations=config['features']['hog_orientations'],
                       pixels_per_cell=config['features']['hog_pixels_per_cell'],
                       cells_per_block=config['features']['hog_cells_per_block'],
                       transform_sqrt=True, block_norm='L2-Hys', visualize=False)
        
        # --- Color Features ---
        hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
        bins = config['features']['color_hist_bins']
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        color_feats = hist.flatten()
        
        # --- Combine and Reshape ---
        combined_feats = np.hstack([hog_feats, color_feats])
        return combined_feats.reshape(1, -1)
    except Exception as e:
        st.error(f"Could not extract features from the image. Error: {e}")
        return None

# ==============================================================================
# UI SIDEBAR
# ==============================================================================
st.sidebar.title("Classifier Controls")

if artifacts:
    # ✨ EXTRA FEATURE 1: Confidence Threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        help="The model will flag predictions with confidence below this value as 'Uncertain'."
    )
    st.sidebar.markdown("---")

    # ✨ EXTRA FEATURE 3: Model Performance Card
    st.sidebar.header("Model Card")
    st.sidebar.info(
        """
        - **Model Type:** RandomForestClassifier
        - **Features:** HOG + Color Histograms
        - **Training Data:** Balanced using data augmentation to improve performance on rare classes.
        """
    )
    class_names = list(label_encoder.classes_)
    with st.sidebar.expander("Show Identifiable Classes"):
        st.write(class_names)
else:
    st.sidebar.error("Model artifacts not loaded.")

# ==============================================================================
# MAIN PAGE INTERFACE
# ==============================================================================
st.title("Advanced Image Classifier")
st.write("Upload an image to classify it using a RandomForest model trained on combined features.")

if not artifacts:
    st.warning("Application is not operational because the model artifacts could not be loaded.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Main layout with two columns
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            with st.spinner('Analyzing image... This may take a moment.'):
                # Run the full pipeline
                features = extract_combined_features(image, config)
                
                if features is not None:
                    # Scale features
                    scaled_features = scaler.transform(features)
                    
                    # Predict using the model
                    prediction_probs = model.predict_proba(scaled_features)
                    confidence_score = np.max(prediction_probs)
                    prediction_index = np.argmax(prediction_probs)
                    predicted_class_name = label_encoder.inverse_transform([prediction_index])[0]

                    st.success("Analysis Complete!")

                    # Display prediction based on confidence threshold
                    if confidence_score >= confidence_threshold:
                        st.metric(label="Predicted Class", value=predicted_class_name.capitalize())
                        st.metric(label="Confidence Score", value=f"{confidence_score:.2%}")
                    else:
                        st.warning(f"Prediction Uncertain")
                        st.metric(label="Predicted Class", value=predicted_class_name.capitalize())
                        st.metric(label="Confidence Score", value=f"{confidence_score:.2%}",
                                  delta=f"Below {confidence_threshold:.0%} threshold", delta_color="inverse")
                    
                    st.markdown("---")

                    # ✨ EXTRA FEATURE 2: What the Model Sees
                    with st.expander("What the Model Sees (Feature Details)"):
                        st.info(f"""
                        The image was converted into a numerical feature vector before classification.
                        - **Total Features:** {features.shape[1]}
                        - **HOG Features:** {config['features']['hog_orientations'] * (config['image_size'][0]//config['features']['hog_pixels_per_cell'][0]) * (config['image_size'][1]//config['features']['hog_pixels_per_cell'][1]) * config['features']['hog_cells_per_block'][0] * config['features']['hog_cells_per_block'][1]}
                        - **Color Features:** {config['features']['color_hist_bins']**3}
                        """)