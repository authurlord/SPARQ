import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total passengers' to numeric (in case of formatting issues)
df['total passengers'] = pd.to_numeric(df['total passengers'], errors='coerce')

# Calculate mean and std
mean_passengers = df['total passengers'].mean()
std_passengers = df['total passengers'].std()

# Define threshold for significant deviation (2 standard deviations)
threshold = 2 * std_passengers
outliers = df[(df['total passengers'] > mean_passengers + threshold) | 
              (df['total passengers'] < mean_passengers - threshold)]

# Extract airport names
outlier_airports = outliers['airport'].tolist()
print(f"Final Answer: {', '.join(outlier_airports)}")