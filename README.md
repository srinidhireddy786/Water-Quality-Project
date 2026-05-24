# AI-Based Water Quality Prediction System

## Project Overview

This project predicts whether water is safe for drinking using Machine Learning.

The system uses water quality parameters such as:
- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

A Random Forest Classifier is trained on the Water Potability Dataset to classify water as:
- Safe water for drinking
- Unsafe water for drinking

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## Machine Learning Workflow

1. Data Ingestion
2. Data Preprocessing
3. Missing Value Handling
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Prediction Pipeline
8. Streamlit Deployment

---

## Project Structure

```text
Water Quality Project/
│
├── artifacts/
├── notebook/
├── src/
│   ├── components/
│   ├── pipelines/
│   └── utils.py
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Model Used

Random Forest Classifier

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## How to Run the Project

### 1. Clone Repository

```bash
git clone <repository-link>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Application

```bash
streamlit run app.py
```

---

## Future Improvements

- IoT Integration
- Real-time Sensor Data
- Multiple ML Models
- Cloud Deployment
- User Authentication

---

## Author

- Srinidhi Reddy (23B81A66J0)
- Sriram Reddy (23B81A05GZ)
