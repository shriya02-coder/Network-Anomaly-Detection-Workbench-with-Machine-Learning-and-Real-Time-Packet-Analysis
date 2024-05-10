# Network Anomaly Detection Workbench with Machine Learning and Real-Time Packet Analysis

The Network Intrusion Detection Workbench is an advanced system that combines machine learning techniques and real-time packet analysis to identify and investigate anomalies in network traffic data. This project aims to provide network security professionals and researchers with a comprehensive tool for detecting potential security threats and unusual patterns in real-time network traffic.

## Features

- **Data Input**: Import packet dataset files or connect to an AWS S3 database to fetch the dataset for analysis.
- **Data Exploration**: Explore and analyze the imported dataset using statistical summaries, visualizations, and data characteristics.
- **Data Preprocessing**: Preprocess the data by handling missing values, applying logarithmic scaling, and selecting custom train-test split ratios.
- **Live Packet Capture**: Perform real-time packet capture and analysis to detect anomalies in live network traffic.
- **Machine Learning Models**: Choose from a wide range of machine learning algorithms, including Naive Bayes, KNN, SVM, Random Forest, Decision Tree, Logistic Regression, Gradient Boosting Classifier, LSTM, and Neural Networks.
- **Model Configuration**: Configure the selected machine learning model's hyperparameters to optimize its performance for specific requirements.
- **Model Evaluation**: Evaluate the trained model's performance using accuracy metrics, classification reports, and various evaluation measures.
- **Anomaly Detection**: Apply the trained model to live packet capture data or imported datasets to identify anomalies and potential security threats in real-time.
- **Metrics Reports**: Generate detailed metrics reports for each classification task, providing insights into the model's performance and detected anomalies.

## Installation

1. Clone the repository: ```git clone https://github.com/your-username/network-intrusion-detection-workbench.git```
2. Install the required dependencies: ```pip install -r requirements.txt```


## Usage
1. Launch the application: ```streamlit run app.py```
2. Access the application through the provided URL in your web browser.
3. Use the intuitive user interface to perform the following tasks:
    - Import packet dataset files.
    - Explore and analyze the imported dataset.
    - Preprocess the data using the available options.
    - Initiate live packet capture sessions and analyze real-time network traffic.
    - Select and configure machine learning models for anomaly detection.
    - Train and evaluate the selected model.
    - Apply the trained model to detect anomalies in live packet capture data or imported datasets.
    - Generate and review metrics reports for each classification task.

## Acknowledgments
I would like to acknowledge the following libraries and frameworks used in this project:

- [Streamlit](https://streamlit.io/) - For building the interactive user interface.
- [scikit-learn](https://scikit-learn.org/) - For machine learning algorithms and evaluation metrics.
- [pandas](https://pandas.pydata.org/) - For data manipulation and analysis.
- [NumPy](https://numpy.org/) - For numerical computing.
- [Matplotlib](https://matplotlib.org/) - For data visualization.

