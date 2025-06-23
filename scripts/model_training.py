import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import load_data, preprocess_data

class DiamondPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(DiamondPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prepare_data(df):
    """Prépare les données pour l'entraînement"""
    X = df.drop('price', axis=1).values
    y = df['price'].values
    
    # Normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Conversion en tenseurs PyTorch
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).view(-1, 1)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, input_size, epochs=100, lr=0.001):
    """Entraîne le modèle"""
    model = DiamondPricePredictor(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

if __name__ == "__main__":
    # Déterminer le dossier courant et le dossier parent
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # Construire les chemins relatifs
    data_path = os.path.join(parent_dir, "data", "diamonds.csv")
    model_path = os.path.join(parent_dir, "models", "diamond_price_predictor.pth")

    # Chargement et prétraitement des données
    data = load_data(data_path)
    data = preprocess_data(data)

    # Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Entraînement du modèle
    input_size = X_train.shape[1]
    model = train_model(X_train, y_train, input_size)

    # Sauvegarde du modèle
    torch.save(model.state_dict(), model_path)
