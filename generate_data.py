"""Generate synthetic housing data for ML training"""
import csv
import random

# Generate synthetic house price data
features = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
locations = ['urban', 'suburban', 'rural']

random.seed(42)
data = []

for _ in range(200):
    sqft = random.randint(500, 4000)
    bedrooms = random.randint(1, 6)
    bathrooms = random.randint(1, 4)
    age = random.randint(0, 50)
    location = random.choice(locations)
    location_score = {'urban': 8, 'suburban': 5, 'rural': 2}[location]
    
    # Price formula: base + sqft*100 + bedrooms*5000 + bathrooms*3000 - age*500 + location bonus
    price = 50000 + sqft * 100 + bedrooms * 5000 + bathrooms * 3000 - age * 500 + location_score * 5000
    price += random.randint(-10000, 10000)  # Add some noise
    
    data.append([sqft, bedrooms, bathrooms, age, location_score, price])

# Write CSV
with open('housing_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(features + ['price'])
    writer.writerows(data)

print("✅ Generated 200 housing records -> housing_data.csv")