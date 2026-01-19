import pandas as pd

df = pd.read_csv('table.csv')
# Extract total passengers as numeric
passenger_counts = df['total passengers'].astype(float)

# Calculate mean and standard deviation
mean_passengers = passenger_counts.mean()
std_passengers = passenger_counts.std()

# Define threshold: more than 2 standard deviations from mean
threshold = 2 * std_passengers
outliers = df[(df['total passengers'].astype(float) < (mean_passengers - threshold)) | 
              (df['total passengers'].astype(float) > (mean_passengers + threshold))]

# Extract airport names of outliers
outlier_airports = outliers['airport'].tolist()

print(f"Final Answer: {', '.join(outlier_airports)}")