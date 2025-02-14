# VisuAIz: Machine Learning Model Visualization App

<!-- TOC -->
* [VisuAIz: Machine Learning Model Visualization App](#visuaiz-machine-learning-model-visualization-app)
  * [Overview](#overview)
  * [Project Structure](#project-structure)
  * [Features](#features)
  * [Installation](#installation)
    * [1. Clone the Repository](#1-clone-the-repository)
    * [2. Create a Virtual Environment (Recommended)](#2-create-a-virtual-environment-recommended)
    * [3. Install Dependencies](#3-install-dependencies)
  * [Running the App](#running-the-app)
  * [Usage](#usage)
  * [Future planning](#future-planning)
<!-- TOC -->

## Overview
Algoviz is a **Streamlit-based** web application that allows users to train and visualize machine learning models for **classification and regression**. The app provides interactive UI elements for model selection, parameter tuning, and visualization of results.
## Project Structure
```
VisuAIz/
│── app.py                  # Main entry point (Navigation & UI handling)
│── classification.py       # Classification logic (ML models & training)
│── regression.py           # Regression logic (ML models & training)
│── visualize.py            # Visualization functions
│── dataset.py              # Dataset loading & preprocessing
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
```

## Features
- Supports **Classification & Regression Models**
- **Dataset Selection**: Load built-in datasets (Iris, Diabetes) or upload custom CSVs
- **Hyperparameter Tuning**: Adjust regularization, kernel types, tree depth, etc.
- **Model Evaluation**: Accuracy, Mean Squared Error (MSE), Confusion Matrix
- **Visualizations**:
- Decision Boundaries (for 2D datasets)
- Confusion Matrices
- Regression Line for single-feature regression

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/kabi-45/Algoviz.git
cd VisuAIz
```

### 2. Create a Virtual Environment (Recommended)
```sh
python -m venv env
env\Scripts\activate     # On Windows
source env/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the App
```sh
streamlit run app.py
```
By default, the app runs on http://localhost:8501.  
This will open the Streamlit app in your default browser.


## Usage
1. **Launch the app** (`streamlit run app.py`)
2. **Navigate** to "_Classification_" or "_Regression_"
3. **Choose dataset & model**, adjust `hyperparameters`
4. **Train & visualize results**

## Future planning
- [ ] Add more algorithms
- [ ] Access to pipelines
- [ ] Add visualization for Trees

---
Created by [kabi-45](https://github.com/kabi-45)