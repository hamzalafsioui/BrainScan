# BrainScan

BrainScan is a deep learning computer vision system designed to classify brain MRI scans into four distinct categories: Glioma, Meningioma, Pituitary tumor, and Normal (No Tumor). The project includes an end to end machine learning pipeline encompassing exploratory data analysis, image preprocessing, convolutional neural network modeling, comprehensive performance evaluation, and an interactive Streamlit web application containerized with Docker.

## Table of Contents

* Overview
* Dataset and Target Classes
* Architecture and Pipeline
* Model Evaluation Results
* Project Directory Structure
* Installation and Local Setup
* Running the Application
* Docker Deployment
* Command Line Inference
* Medical Disclaimer

## Overview

Brain tumor identification and classification from Magnetic Resonance Imaging (MRI) plays a critical role in diagnostic neurology. BrainScan provides an automated classification tool capable of analyzing brain MRI slices, generating category predictions, and displaying confidence scores across all supported diagnostic classes.

## Dataset and Target Classes

The model is trained to distinguish between four distinct classes:

* Glioma: Tumors originating in the glial cells of the brain or spinal cord.
* Meningioma: Tumors developing in the meninges, the protective membrane surrounding the brain.
* No Tumor: Healthy brain MRI scans with no tumor present.
* Pituitary: Tumors developing in the pituitary gland at the base of the brain.

Input images are processed in BGR color format and resized to 224 by 224 pixels with three color channels.

## Architecture and Pipeline

The BrainScan CNN architecture is designed for balance between representation capacity and inference speed:

1. Input Layer: Accepts RGB/BGR image tensors of shape (224, 224, 3).
2. Convolutional Block 1: 32 filters (3x3 kernel, ReLU activation), Batch Normalization, MaxPooling (2x2).
3. Convolutional Block 2: 64 filters (3x3 kernel, ReLU activation), Batch Normalization, MaxPooling (2x2).
4. Convolutional Block 3: 128 filters (3x3 kernel, ReLU activation), Batch Normalization, MaxPooling (2x2), Dropout (0.25).
5. Convolutional Block 4: 128 filters (3x3 kernel, ReLU activation), Batch Normalization, MaxPooling (2x2), Dropout (0.25).
6. Classifier Head: Flatten layer (25088 features), Dense layer with 128 units (ReLU activation), Dropout (0.50), Final Dense output layer with 4 units (Softmax activation).

Training settings:
* Optimizer: Adam (learning rate 1e-4)
* Loss Function: Categorical Crossentropy
* Metric: Accuracy
* Batch Size: 32

## Model Evaluation Results

Evaluation performed on an independent test split of 1,405 MRI images:

* Overall Test Accuracy: 95.37%
* Macro Average Precision: 95.28%
* Macro Average Recall: 95.14%
* Macro Average F1 Score: 95.19%
* Weighted Average F1 Score: 95.37%

Per Class Performance Breakdown:

* Glioma:
  * Precision: 96.12%
  * Recall: 91.67%
  * F1 Score: 93.84%
  * Test Samples: 324

* Meningioma:
  * Precision: 90.21%
  * Recall: 92.40%
  * F1 Score: 91.29%
  * Test Samples: 329

* No Tumor:
  * Precision: 97.05%
  * Recall: 98.75%
  * F1 Score: 97.89%
  * Test Samples: 400

* Pituitary:
  * Precision: 97.73%
  * Recall: 97.73%
  * F1 Score: 97.73%
  * Test Samples: 352

## Project Directory Structure

* app/
  * streamlit_app.py: Web interface for interactive MRI image analysis.
* data/
  * raw/: Raw input MRI dataset organized by class folders.
  * processed/: Preprocessed image arrays stored in npz format.
* models/
  * best_brainscan.keras: Trained convolutional neural network weights and architecture.
  * class_names.json: Ordered class labels mapping.
  * training_history.json: Epoch by epoch training loss and accuracy logs.
  * model_architecture.png: Visual diagram of neural network layers.
* notebooks/
  * 01_data_exploration.ipynb: Dataset inspection, distribution, and pixel statistics.
  * 02_preprocessing_and_model.ipynb: Image normalization, data splitting, and CNN baseline.
  * 03_training_model_kaggle.ipynb: Training workflow executed on GPU accelerators.
  * 04_evaluation.ipynb: Model evaluation, confusion matrix, and visual error analysis.
* reports/
  * confusion_matrix.png: Normalized confusion matrix heatmap.
  * learning_curves.png: Training versus validation loss and accuracy curves.
  * test_metrics.json: Quantitative evaluation metrics on test dataset.
  * correct_predictions.png: Visual gallery of successful classifications.
  * wrong_predictions.png: Error analysis showing misclassified cases.
* src/
  * config.py: Centralized configuration parameters and constants.
  * data_loader.py: Dataset directory scanner and image loader.
  * eda.py: Exploratory data analysis visualization utilities.
  * evaluate.py: Model evaluation metrics, charts, and reporting routines.
  * model.py: Keras model architecture definition and compilation.
  * predict.py: Single image and batch inference routines.
  * preprocessing.py: Resizing, normalization, and label encoding utilities.
  * utility.py: Helper functions for plotting and file handling.
* test_images/: Sample MRI scans for quick verification.
* Dockerfile: Production container definition for Streamlit.
* docker-compose.yml: Orchestration specification for single command container launch.
* requirements.txt: Complete environment dependencies.
* requirements-app.txt: Lightweight production dependencies for the web container.

## Installation and Local Setup

1. Clone or download the repository to your local machine:
   cd BrainScan

2. Create and activate a Python virtual environment:
   python -m venv .venv
   Windows: .venv\Scripts\activate
   Linux or macOS: source .venv/bin/activate

3. Install required dependencies:
   pip install -r requirements.txt

## Running the Application

Launch the Streamlit web dashboard locally:

   streamlit run app/streamlit_app.py

After launching, open your browser and navigate to:
   http://localhost:8501

Upload any JPEG or PNG brain MRI scan to inspect predicted tumor type and confidence percentages.

## Docker Deployment

To build and run the application container using Docker Compose:

1. Start the container in detached mode:
   docker compose up -d

2. Access the application in your browser:
   http://localhost:8501

3. Stop the container when finished:
   docker compose down

The Docker image uses a multi step caching design where dependencies in requirements-app.txt are cached independently from application source code. Subsequent builds skip package installation and complete in seconds.

## Command Line Inference

You can also run predictions directly from the terminal using src/predict.py:

   python src/predict.py test_images/brain_glioma_0001.jpg

Output displays the predicted diagnostic category, confidence percentage, and the full probability distribution across all four classes.

## Medical Disclaimer

This software is an educational prototype developed for machine learning research and demonstration purposes only. It is neither approved as a medical device nor intended to replace professional diagnostic judgment by qualified physicians or radiologists.

