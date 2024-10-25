Stock Price Prediction Using LSTM
This repository contains a time series forecasting model that predicts stock prices using a Long Short-Term Memory (LSTM) neural network in TensorFlow/Keras. The project includes data preprocessing, model training, evaluation, and visualization of predictions against actual values.

Table of Contents
Project Overview
Dataset
Preprocessing
Model Architecture
Training
Evaluation
Results
Usage
Dependencies
License
Project Overview
This project demonstrates the use of an LSTM neural network to predict stock closing prices based on features including open, high, low, and volume. It uses callbacks like early stopping and learning rate reduction to optimize training performance and prevent overfitting.

Dataset
The dataset includes stock market data with features:

open: Opening price
high: Highest price
low: Lowest price
close: Closing price (target variable)
volume: Volume of stocks traded
Ensure the data file train-00000-of-00001-0866b62cb4c6a541.csv is in your working directory.

Preprocessing
Missing Values: Drops rows with missing values.
Feature Scaling: Uses MinMax scaling to normalize the input features and target variable.
Data Reshaping: Uses 60 time steps to create sequences for input to the LSTM.
Model Architecture
The LSTM model includes:

Two LSTM layers with 50 units each and dropout layers for regularization.
Dense layers for final output.
Huber loss function and Adam optimizer with gradient clipping for stable training.
python

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.4))
model.add(LayerNormalization())
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.4))
model.add(LayerNormalization())
model.add(Dense(units=25))
model.add(Dense(units=1))
Training
The model trains with early stopping and learning rate reduction callbacks for improved performance:


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
Evaluation
Metrics used for model evaluation include:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R-squared Score (R²)
Results
The model's performance on the test set:

Mean Absolute Error: 36216.4352
Root Mean Squared Error: 924045.2613
R-squared Score: 0.6878
Usage
To run the model, clone the repository, install the dependencies, and ensure the dataset file is in the correct path. Then, execute the following script:


python stock_price_prediction.py
Dependencies
Python 3.x
TensorFlow/Keras
NumPy
Pandas
Matplotlib
Scikit-Learn
Install the dependencies using:

pip install -r requirements.txt
