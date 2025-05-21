import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Step 1: Data Collection
def download_stock_data(ticker, start_date, end_date, csv_file):
    data = yf.download(ticker, start=start_date, end=end_date)
    # Save all relevant columns and reset index
    data = data.reset_index()
    data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv(csv_file, index=False)
    return data

def add_technical_indicators(data):
    # Calculate moving averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    
    # Calculate Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (std * 2)
    data['BB_lower'] = data['BB_middle'] - (std * 2)
    
    # Calculate price changes
    data['Price_Change'] = data['Close'].pct_change()
    data['Price_Change_5'] = data['Close'].pct_change(periods=5)
    
    # Calculate volatility
    data['Volatility'] = data['Close'].rolling(window=10).std()
    
    return data

# Step 2: Data Preprocessing
def preprocess_data(seq_length):
    # Read the CSV file with proper date parsing
    data = pd.read_csv("./ITC_NS_stock_data.csv", parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    
    # Convert all columns to numeric first
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop any rows with NaN values
    data = data.dropna()
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Convert volume to billions to normalize it
    data['Volume'] = data['Volume'] / 1e9
    
    # Drop any remaining NaN values from technical indicators
    data = data.dropna()

    # Scale all features
    all_features_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = all_features_scaler.fit_transform(data)
    
    # Create a separate scaler for Close prices
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(data[['Close']])

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, data.shape[1]-1])  # Use last column (Close)
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Convert to PyTorch tensors and move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    return X_train, y_train, X_test, y_test, close_scaler, all_features_scaler, data, device

# Step 3: Model Building
class StockPredictor(nn.Module):
    def __init__(self, seq_length, n_features):
        super(StockPredictor, self).__init__()
        
        # Efficient CNN layers
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Calculate the size after CNN layers
        cnn_output_size = ((seq_length - 4) // 2)
        
        # GRU layer (faster than LSTM)
        self.gru = nn.GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Linear(128, 1)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape input for CNN
        x = x.transpose(1, 2)
        
        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Reshape for GRU
        x = x.transpose(1, 2)
        
        # GRU layer
        x, _ = self.gru(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attention_weights, dim=1)
        
        # Dense layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        return x

# Step 4: Training the Model
def train_model(model, X_train, y_train, epochs, batch_size, device):
    criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Lower learning rate, less regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True, min_lr=1e-6)  # Gentler LR reduction
    
    # Create DataLoader with pin_memory for faster data transfer to GPU
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    history = []
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 15  # Increased patience
    best_model_state = None
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced gradient clipping
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping with best model saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f'Early stopping at epoch {epoch+1}')
            print(f'Best loss: {best_loss:.6f}')
            model.load_state_dict(best_model_state)  # Restore best model
            break
            
        if (epoch + 1) % 5 == 0:  # Print more frequently
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}')
    
    return history

# Step 5: Evaluation
def evaluate_model(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        
    # Convert predictions back to original scale
    predictions = predictions.numpy()
    predictions = scaler.inverse_transform(predictions)
    y_test = y_test.numpy().reshape(-1, 1)
    y_test = scaler.inverse_transform(y_test)

    rmse = np.sqrt(np.mean((predictions - y_test)**2))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')

    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual Price', alpha=0.8)
    plt.plot(predictions, label='Predicted Price', alpha=0.8)
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 6: Future Prediction
def future_prediction(model, data, close_scaler, all_features_scaler, seq_length):
    model.eval()
    with torch.no_grad():
        # Get the last sequence
        last_sequence = data[-seq_length:].values
        # Transform using the all features scaler
        last_sequence = all_features_scaler.transform(last_sequence)
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)
        
        # Get prediction
        prediction = model(last_sequence)
        prediction = prediction.numpy()
        # Use close_scaler for inverse transform of the predicted close price
        prediction = close_scaler.inverse_transform(prediction)
        
        # Get recent prices for display
        recent_dates = data.index[-10:]  # Last 10 days
        recent_prices = data['Close'][-10:]  # Last 10 days of closing prices
        
        # Create prediction date (today)
        today = pd.Timestamp.now().normalize()
        today_str = today.strftime('%Y-%m-%d')  # Convert to string for plotting
        
        # Plot recent trend and prediction
        plt.figure(figsize=(12, 6))
        plt.plot(recent_dates, recent_prices, marker='o', label='Recent Prices')
        plt.plot([recent_dates[-1], today_str], [recent_prices[-1], prediction[0][0]], 
                 'r--', marker='o', label='Today\'s Prediction')
        plt.title('Recent Price Trend and Today\'s Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Display prediction with relevant information
        print("\nPrediction Summary:")
        print("-" * 50)
        print(f"Last Closing Price: ₹{recent_prices[-1]:.2f}")
        print(f"Predicted Price for Today ({today_str}): ₹{prediction[0][0]:.2f}")
        change = ((prediction[0][0] - recent_prices[-1]) / recent_prices[-1]) * 100
        print(f"Expected Change: {change:+.2f}%")
        print("-" * 50)
        
        return prediction[0][0]

# Step 7: Save and Load Model
def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def load_saved_model(file_name, seq_length):
    model = StockPredictor(seq_length, n_features=5)  # 5 features: Open, High, Low, Close, Volume
    model.load_state_dict(torch.load(file_name))
    return model

# Main Function
def main():
    ticker = 'ITC.NS'
    start_date = '2010-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')  # Use today's date
    csv_file = 'ITC_NS_stock_data.csv'
    seq_length = 20  # Further reduced sequence length for faster training
    epochs = 150  # Increased epochs
    batch_size = 128  # Increased batch size for better stability
    model_file = 'stock_prediction_model.pth'

    print("\nDownloading latest stock data...")
    # Download stock data
    # data = download_stock_data(ticker, start_date, end_date, csv_file)

    print("Processing data and training model...")
    # Preprocess data
    X_train, y_train, X_test, y_test, close_scaler, all_features_scaler, data, device = preprocess_data( seq_length)

    # Get number of features
    n_features = X_train.shape[2]

    # Build model with correct input features and move to device
    model = StockPredictor(seq_length, n_features).to(device)

    # Train model
    history = train_model(model, X_train, y_train, epochs, batch_size, device)

    print("\nAnalyzing model performance...")
    # Plot training loss
    plt.figure(figsize=(14, 7))
    plt.plot(history, label='Training Loss')
    plt.title('Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Move model and data to CPU for evaluation
    model = model.cpu()
    X_test = X_test.cpu()
    y_test = y_test.cpu()

    # Evaluate model
    print("\nModel Evaluation Results:")
    evaluate_model(model, X_test, y_test, close_scaler)

    print("\nGenerating tomorrow's prediction...")
    # Future prediction
    predicted_price = future_prediction(model, data, close_scaler, all_features_scaler, seq_length)

    # Save model
    save_model(model, model_file)
    print(f"\nModel saved as {model_file}")

if __name__ == "__main__":
    main()
