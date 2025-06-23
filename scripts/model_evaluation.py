import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from model_training import DiamondPricePredictor, prepare_data
from data_processing import load_data, preprocess_data

def evaluate_model(model_path, data_path):
    """Évalue le modèle sur les données de test"""
    # Chargement des données
    data = load_data(data_path)
    data = preprocess_data(data)
    
    # Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Chargement du modèle
    model = DiamondPricePredictor(X_train.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Force le chargement sur CPU
    model.eval()
    
    # Assure que toutes les opérations sont sur CPU
    device = torch.device('cpu')
    model = model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Prédictions
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()  # Conversion explicite en numpy array
    
    # Conversion des tenseurs en numpy arrays si nécessaire
    if torch.is_tensor(y_test):
        y_test = y_test.cpu().numpy()
    
    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Scor.e.: {r2:.2f}")
    
    return mse, rmse, r2

if __name__ == "__main__":
    evaluate_model('C:/Users/COMPUTER-STORE/Documents/Environnement/ANN/diamond_ana/models/diamond_price_predictor.pth', 'C:/Users/COMPUTER-STORE/Documents/Environnement/ANN/diamond_ana/data/diamonds.csv')