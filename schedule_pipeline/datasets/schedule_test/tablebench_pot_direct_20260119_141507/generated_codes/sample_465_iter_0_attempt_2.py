import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total passengers' to numeric (in case of non-numeric strings)
df['total passengers'] = pd.to_numeric(df['total passengers'], errors='coerce')

# Calculate mean and std
mean_passengers = df['total passengers'].mean()
std_passengers = df['total passengers'].std()

# Define threshold for outliers (more than 2 standard deviations from mean)
lower_bound = mean_passengers - 2 * std_passengers
upper_bound = mean_passengers + 2 * std_passengers

# Find airports outside this range
outliers = df[(df['total passengers'] < lower_bound) | (df['total passengers'] > upper_bound)]
outlier_airports = outliers['airport'].tolist()

print(f"Final Answer: {', '.join(outlier_airports)}")