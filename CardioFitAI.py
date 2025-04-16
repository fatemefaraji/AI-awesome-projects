"""
HeartRiskPredictor: Predicting Heart Disease Risk from Biking and Smoking

This project builds a deep neural network to predict heart disease risk based on
biking and smoking data. It includes data preprocessing, an optimized MLP model
with dropout and batch normalization, and comprehensive evaluation metrics.
Visualizations help interpret model performance and data relationships.

Author: Adapted and enhanced from Sreenivas Bhattiprolu's code
License: Free to use with acknowledgment
Video Reference: https://youtu.be/j2kfzYR_abI
Dataset: https://cdn.scribbr.com/wp-content/uploads//2020/02/heart.data_.zip

Dependencies: tensorflow, pandas, numpy, seaborn, matplotlib, scikit-learn
"""

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configure logging for debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_PATH = "data/heart_data.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_data(file_path):
    """Load and preprocess the heart disease dataset."""
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully")
        
        # Drop unnecessary columns
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        
        # Check for missing values
        if df.isnull().sum().any():
            logger.warning("Missing values detected. Filling with mean.")
            df.fillna(df.mean(), inplace=True)
        
        # Extract features and target
        X = df[['biking', 'smoking']].values
        y = df['heart.disease'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler, df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def visualize_data(df):
    """Create visualizations to understand data relationships."""
    plt.figure(figsize=(12, 5))
    
    # Biking vs Heart Disease
    plt.subplot(1, 2, 1)
    sns.regplot(x='biking', y='heart.disease', data=df, line_kws={'color': 'red'})
    plt.title('Biking vs Heart Disease')
    
    # Smoking vs Heart Disease
    plt.subplot(1, 2, 2)
    sns.regplot(x='smoking', y='heart.disease', data=df, line_kws={'color': 'red'})
    plt.title('Smoking vs Heart Disease')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_relationships.png'))
    plt.show()

def build_model(input_dim=2):
    """Build and compile a deep neural network model."""
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Linear activation for regression
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    logger.info("Model summary:")
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=500):
    """Train the model with early stopping and checkpointing."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance and print metrics."""
    predictions = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"Evaluation Metrics:")
    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Heart Disease Risk')
    plt.ylabel('Predicted Heart Disease Risk')
    plt.title('Actual vs Predicted Heart Disease Risk')
    plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'))
    plt.show()
    
    return predictions, mse, mae, r2

def plot_training_history(history):
    """Plot training and validation loss and MAE."""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.show()

def print_model_weights(model):
    """Print model weights and biases for interpretability."""
    for layer_idx, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            weights, biases = layer.get_weights()
            logger.info(f"___________ Layer {layer_idx} __________")
            for neuron_idx, bias in enumerate(biases):
                logger.info(f"Bias to Layer{layer_idx+1}Neuron{neuron_idx}: {bias:.6f}")
            for from_neuron, wgt in enumerate(weights):
                for to_neuron, wgt_val in enumerate(wgt):
                    logger.info(f"Layer{layer_idx}, Neuron{from_neuron} to Layer{layer_idx+1}, Neuron{to_neuron} = {wgt_val:.6f}")

def predict_single(model, scaler, biking, smoking):
    """Predict heart disease risk for a single input."""
    input_data = scaler.transform([[biking, smoking]])
    prediction = model.predict(input_data)[0][0]
    logger.info(f"Predicted Heart Disease Risk for biking={biking}, smoking={smoking}: {prediction:.2f}")
    return prediction

def main():
    """Main function to run the heart disease prediction pipeline."""
    logger.info("Starting HeartRiskPredictor Pipeline")
    
    # Load and preprocess data
    X, y, scaler, df = load_and_preprocess_data(DATA_PATH)
    
    # Visualize data
    visualize_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Build and train model
    model = build_model(input_dim=X.shape[1])
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    predictions, mse, mae, r2 = evaluate_model(model, X_test, y_test, scaler)
    
    # Plot training history
    plot_training_history(history)
    
    # Print model weights
    print_model_weights(model)
    
    # Example prediction
    sample_biking = 65.1292
    sample_smoking = 2.21956
    predict_single(model, scaler, sample_biking, sample_smoking)
    
    logger.info("HeartRiskPredictor Pipeline Completed")

if __name__ == "__main__":
    main()