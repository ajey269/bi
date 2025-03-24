linear regression

x <- c(151, 174, 138, 186, 128, 136, 179, 163, 152, 131)
y <- c(63, 81, 56, 91, 47, 57, 76, 72, 62, 48)
# Apply the lm() function to create a linear regression model
relation <- lm(y ~ x)
# Print the model summary
print(summary(relation))
# Predict the weight of a person with height 170
a <- data.frame(x = 170)
result <- predict(relation, a)
print(result)
# Visualizing the Regression Graphically
png(file = "linearregression.png")
plot(x, y, col = "blue", main = "Height & Weight Regression",
 xlab = "Height in cm", ylab = "Weight in Kg", pch = 16)
abline(relation, col = "red")
dev.off()


//////////
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Height (x) and Weight (y) data
x = np.array([151, 174, 138, 186, 128, 136, 179, 163, 152, 131]).reshape(-1, 1)
y = np.array([63, 81, 56, 91, 47, 57, 76, 72, 62, 48])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Print the model summary (Coefficient and Intercept)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict the weight of a person with height 170 cm
height_170 = np.array([[170]])
predicted_weight = model.predict(height_170)
print("Predicted weight for height 170 cm:", predicted_weight[0])

# Visualizing the Regression Graphically
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="blue", label="Actual Data", marker="o")
plt.plot(x, model.predict(x), color="red", label="Regression Line")

# Graph Labels
plt.title("Height & Weight Regression")
plt.xlabel("Height in cm")
plt.ylabel("Weight in Kg")
plt.legend()

# Save the plot as a PNG file
plt.savefig("linearregression.png")
plt.show()
