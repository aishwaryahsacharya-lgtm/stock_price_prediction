import numpy as np
from sklearn.linear_model import LinearRegression

# Historical card
prices = [100, 102, 101, 105, 107, 110]

# Prepare dataset
x = np.array(prices[:-1]).reshape(-1, 1)
y = np.array(prices[1:])

# Create and train the model
model = LinearRegression()
model.fit(x, y)

# Predict next day's stock price
prediction = model.predict([[121]])
print("Predicted next stock price:", prediction)
