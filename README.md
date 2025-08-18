# Machine_failure_prediction_using_MachineLearning

# Machine Failure Prediction

This repository contains a machine learning pipeline and deployment app for predicting machine failures based on sensor data (e.g., air temperature, process temperature,rotational speed). The project uses a **Random Forest Classifier** trained on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository.

---

## Repository Structure

- **`ai4i2020.csv`**  
  The dataset adopted from the UCI Machine Learning Repository and used to train the model.  

- **`machine_failure_prediction.ipynb`**  
  Jupyter Notebook containing the full training pipeline:  
  - Data preprocessing (dropping unnecessary columns, one-hot encoding)  
  - Model training with RandomizedSearchCV  
  - Evaluation (confusion matrix, classification report)  
  - Saving the final trained model  

- **`failure_prediction_model.joblib`**  
  Serialized Random Forest model trained on the AI4I dataset. Can be directly loaded for inference.  

- **`app.py`**  
  Streamlit app that loads the trained model and provides an interactive web interface for predicting machine failure from user inputs.  

- **`requirements.txt`**  
  List of Python dependencies needed to run the training and app.  


