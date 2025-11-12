# Waste Classification App

This is a Streamlit web application that uses a deep learning model to classify images of waste into four categories: metal, organic, paper, and plastic.

## Features

-   **Image Classification:** Classifies waste images using a pre-trained Keras model (`waste_classification_mobilenetv2_pro.keras`).
-   **Multiple Input Methods:** Supports both file uploads and direct camera input for image classification.
-   **Interactive Web UI:** Built with Streamlit for an easy-to-use interface.

## Project Structure

```
.
├── images/
│   ├── test/
│   └── train/
├── venv/
├── app.py
├── best_waste_model.keras
├── requirements.txt
├── train.py
├── train2.py
├── train3.py
├── waste_classification_mobilenetv2.keras
└── waste_classification_mobilenetv2_pro.keras
```

## Setup and Installation

### Prerequisites

-   Python 3.9+
-   A virtual environment tool (like `venv`)

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a virtual environment:**
    On Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    On macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```
Or, to be sure you are using the correct python environment:
```bash
python -m streamlit run app.py
```

This will open the application in your default web browser.

## How to Use the Application

1.  **Choose an input method:**
    -   Select "Upload a file" to upload an image from your computer.
    -   Select "Use camera" to take a new picture using your device's camera.

2.  **Provide an image:**
    -   If uploading, click "Browse files" and select an image.
    -   If using the camera, click "Take a picture".

3.  **View the prediction:**
    The application will display the image and the predicted waste category along with a confidence score.
