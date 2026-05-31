"""House Price Prediction - Linear Regression with Normalization"""
import csv
import random
import math

# Load data
def load_data(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row['sqft']), float(row['bedrooms']), float(row['bathrooms']), 
                     float(row['age']), float(row['location_score'])])
            y.append(float(row['price']))
    return X, y

# Normalize features
def normalize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = [math.sqrt(sum((X[i][j] - means[j])**2 for i in range(len(X))) / len(X)) for j in range(n_features)]
    stds = [s if s > 0 else 1 for s in stds]  # Avoid division by zero
    
    X_norm = []
    for x in X:
        X_norm.append([(x[j] - means[j]) / stds[j] for j in range(n_features)])
    return X_norm, means, stds

# Linear Regression with normalization
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0
        self.means = None
        self.stds = None
    
    def fit(self, X, y, lr=0.1, epochs=500):
        # Normalize first
        X_norm, self.means, self.stds = normalize(X)
        
        n_features = len(X_norm[0])
        self.weights = [0.0] * n_features
        self.bias = sum(y) / len(y)  # Start with mean
        n = len(X_norm)
        
        for epoch in range(epochs):
            predictions = [sum(w * xi for w, xi in zip(self.weights, x)) + self.bias for x in X_norm]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            
            # Gradient descent
            for j in range(n_features):
                grad = sum(errors[i] * X_norm[i][j] for i in range(n)) / n
                self.weights[j] -= lr * grad
            
            bias_grad = sum(errors) / n
            self.bias -= lr * bias_grad
            
            if epoch % 100 == 0:
                mse = sum(e**2 for e in errors) / n
                print(f"   Epoch {epoch}: MSE = {mse:,.0f}")
    
    def predict(self, x):
        # Normalize input
        x_norm = [(x[j] - self.means[j]) / self.stds[j] for j in range(len(x))]
        return sum(w * xi for w, xi in zip(self.weights, x_norm)) + self.bias

# Main
if __name__ == '__main__':
    # Generate synthetic data
    print("📊 Generating housing data...")
    exec(open('generate_data.py').read())
    
    # Load data
    X, y = load_data('housing_data.csv')
    
    # Split: 80% train, 20% test
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    print(f"📈 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    model = LinearRegression()
    print("🤖 Training model...")
    model.fit(X_train, y_train, lr=0.1, epochs=500)
    
    # Evaluate
    predictions = [model.predict(x) for x in X_test]
    mse = sum((p - y)**2 for p, y in zip(predictions, y_test)) / len(X_test)
    rmse = math.sqrt(mse)
    mae = sum(abs(p - y) for p, y in zip(predictions, y_test)) / len(X_test)
    
    print(f"\n📊 Results:")
    print(f"   RMSE: ${rmse:,.0f}")
    print(f"   MAE:  ${mae:,.0f}")
    
    # Demo prediction
    print(f"\n🏠 Sample Predictions:")
    samples = [
        [1500, 3, 2, 10, 8],    # Urban house
        [2000, 4, 3, 5, 5],     # Suburban house
        [800, 2, 1, 30, 2],     # Rural house
    ]
    loc_names = {8: 'urban', 5: 'suburban', 2: 'rural'}
    for house in samples:
        price = model.predict(house)
        loc = loc_names.get(house[4], 'unknown')
        print(f"   {house[0]}sqft, {house[1]}bed/{house[2]}bath, {house[3]}yr old ({loc}) -> ${price:,.0f}")
    
    print("\n✅ Model trained successfully!")