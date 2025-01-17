# CreditCard_Fraud_Detection
# Credit Card Fraud Detection using Random Forest

## Overview
This repository contains a project for detecting credit card fraud using a Random Forest classifier. The model is trained to distinguish between legitimate transactions (label 0) and fraudulent transactions (label 1) using a dataset of 27,360 samples. The classifier achieves high performance, as detailed below.

## Dataset
- **Total Samples:** 27,360
- **Legitimate Transactions (label 0):** 27,297
- **Fraudulent Transactions (label 1):** 63
- The dataset is highly imbalanced, with a very small proportion of fraudulent transactions.

## Model Performance
### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.00      | 1.00   | 1.00     | 27,297  |
| 1.0   | 0.95      | 0.84   | 0.89     | 63      |

**Accuracy:** 1.00  
**Macro Average:** Precision: 0.97, Recall: 0.92, F1-Score: 0.95  
**Weighted Average:** Precision: 1.00, Recall: 1.00, F1-Score: 1.00  

### ROC-AUC Score
The model achieved a **ROC-AUC score of 0.964**, indicating strong discriminatory power between legitimate and fraudulent transactions.

## Handling Imbalance
Given the highly imbalanced nature of the dataset, the following techniques were employed to improve model performance:
- **Oversampling:** Synthetic Minority Oversampling Technique (SMOTE) was used to balance the dataset by generating synthetic samples for the minority class.
- **Cost-sensitive Learning:** Adjusted the class weights in the Random Forest model to penalize misclassification of the minority class.

## Future Prediction
The trained model can be used to predict fraudulent transactions on new, unseen data. Simply load the saved model and provide transaction data in the specified format. Example usage:
```python
from src.predict import load_model, predict

model = load_model('models/random_forest_model.pkl')
new_data = [...]  # Replace with your transaction data
predictions = predict(model, new_data)
print(predictions)
```

## Future Improvements
To further enhance the performance and robustness of the model, the following improvements are planned:
- **Feature Engineering:** Explore advanced techniques to create more meaningful features from the transaction data.
- **Ensemble Learning:** Combine predictions from multiple models to improve accuracy and stability.
- **Real-time Prediction:** Implement a pipeline for real-time fraud detection with minimal latency.
- **Explainability:** Use tools like SHAP or LIME to provide insights into model predictions.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the dataset into the `data/` directory.
2. Run the Jupyter notebook or Python scripts to preprocess the data and train the model.
   ```bash
   python train_model.py
   ```
3. Evaluate the model performance using the included evaluation script:
   ```bash
   python evaluate_model.py
   ```

## Features
- **Data Preprocessing:** Handles imbalanced data using techniques such as oversampling and feature scaling.
- **Model Training:** Trains a Random Forest classifier to identify fraudulent transactions.
- **Performance Evaluation:** Provides detailed classification reports and ROC-AUC metrics.
- **Imbalance Handling:** Implements techniques like SMOTE and cost-sensitive learning to address class imbalance.
- **Future Prediction:** Supports prediction of fraudulent transactions on new data.

## File Structure
- `data/`: Directory to store the dataset.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and model development.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `models/`: Saved models for reuse.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
Special thanks to the open-source community and the creators of the dataset for making this work possible.

---
Feel free to contribute by opening issues or submitting pull requests!

