"""House Price Prediction Web App"""
from flask import Flask, request, render_template_string
import csv
import math
import pickle
import os

app = Flask(__name__)

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

def normalize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = [math.sqrt(sum((X[i][j] - means[j])**2 for i in range(len(X))) / len(X)) for j in range(n_features)]
    stds = [s if s > 0 else 1 for s in stds]
    X_norm = []
    for x in X:
        X_norm.append([(x[j] - means[j]) / stds[j] for j in range(n_features)])
    return X_norm, means, stds

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0
        self.means = None
        self.stds = None
    
    def fit(self, X, y, lr=0.1, epochs=500):
        X_norm, self.means, self.stds = normalize(X)
        n_features = len(X_norm[0])
        self.weights = [0.0] * n_features
        self.bias = sum(y) / len(y)
        n = len(X_norm)
        
        for _ in range(epochs):
            predictions = [sum(w * xi for w, xi in zip(self.weights, x)) + self.bias for x in X_norm]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            
            for j in range(n_features):
                grad = sum(errors[i] * X_norm[i][j] for i in range(n)) / n
                self.weights[j] -= lr * grad
            
            bias_grad = sum(errors) / n
            self.bias -= lr * bias_grad
    
    def predict(self, x):
        x_norm = [(x[j] - self.means[j]) / self.stds[j] for j in range(len(x))]
        return sum(w * xi for w, xi in zip(self.weights, x_norm)) + self.bias

# Train model on startup
print("🤖 Loading & training model...")
X, y = load_data('housing_data.csv')
model = LinearRegression()
model.fit(X, y, lr=0.1, epochs=500)
print("✅ Model ready!")

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>🏠 House Price Predictor</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e); min-height: 100vh; margin: 0; display: flex; align-items: center; justify-content: center; }
        .container { background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); width: 400px; }
        h1 { color: #1a1a2e; margin-bottom: 1.5rem; text-align: center; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; color: #555; margin-bottom: 0.5rem; font-weight: 500; }
        input, select { width: 100%; padding: 0.75rem; border: 2px solid #eee; border-radius: 8px; font-size: 1rem; box-sizing: border-box; }
        input:focus, select:focus { outline: none; border-color: #667eea; }
        button { width: 100%; padding: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); }
        .result { margin-top: 1.5rem; padding: 1rem; background: linear-gradient(135deg, #f093fb, #f5576c); border-radius: 8px; text-align: center; color: white; display: none; }
        .result.show { display: block; animation: fadeIn 0.5s; }
        .price { font-size: 2rem; font-weight: bold; }
        @keyframes fadeIn { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 House Price Predictor</h1>
        <form method="POST">
            <div class="form-group">
                <label>Square Feet</label>
                <input type="number" name="sqft" placeholder="e.g. 1500" required>
            </div>
            <div class="form-group">
                <label>Bedrooms</label>
                <input type="number" name="bedrooms" placeholder="e.g. 3" required>
            </div>
            <div class="form-group">
                <label>Bathrooms</label>
                <input type="number" name="bathrooms" placeholder="e.g. 2" required>
            </div>
            <div class="form-group">
                <label>Age (years)</label>
                <input type="number" name="age" placeholder="e.g. 10" required>
            </div>
            <div class="form-group">
                <label>Location</label>
                <select name="location">
                    <option value="8">Urban</option>
                    <option value="5">Suburban</option>
                    <option value="2">Rural</option>
                </select>
            </div>
            <button type="submit">Predict Price</button>
        </form>
        {% if price %}
        <div class="result show">
            <div>Estimated Price</div>
            <div class="price">${{ price }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    price = None
    if request.method == 'POST':
        sqft = int(request.form['sqft'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        age = int(request.form['age'])
        location = int(request.form['location'])
        
        features = [sqft, bedrooms, bathrooms, age, location]
        price = model.predict(features)
        price = f"{int(price):,}"
    
    return render_template_string(HTML, price=price)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)