// Perform the data classification using classification algorithm using
R/Python.



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define rainfall data
rainfall = [799, 1174.8, 865.1, 1334.6, 635.4, 918.5, 685.5, 998.6, 784.2, 985, 882.8, 1071]

# Create a date range starting from January 2012 with a monthly frequency
dates = pd.date_range(start="2012-01", periods=len(rainfall), freq="M")

# Create a Pandas Series for time series data
rainfall_series = pd.Series(rainfall, index=dates)

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(rainfall_series, marker="o", linestyle="-", color="b", label="Rainfall (mm)")
plt.title("Monthly Rainfall Time Series")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.legend()

# Save the plot as a PNG file
plt.savefig("rainfall.png")
plt.show()  # Show the plot (optional)






//////////

rainfall <-c(799,1174.8,865.1,1334.6,635.4,918.5,685.5,998.6,784.2,985,882.8,1071)
rainfall.timeseries <- ts(rainfall,start = c(2012,1),frequency = 12)
print(rainfall.timeseries)
png(file = "rainfal.png")
plot(rainfall.timeseries)
dev.off()

