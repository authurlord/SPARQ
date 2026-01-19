import pandas as pd

df = pd.read_csv('table.csv')

# Check for unusually high total passengers and annual change
# Convert 'total passengers' and 'annual change' to numeric
df['total passengers'] = pd.to_numeric(df['total passengers'], errors='coerce')
df['annual change'] = pd.to_numeric(df['annual change'].str.replace('%', ''), errors='coerce')

# Find the row with maximum total passengers and highest annual change
max_passengers = df['total passengers'].max()
max_annual_change = df['annual change'].max()

# Find the city with both high values
outlier_row = df[(df['total passengers'] == max_passengers) & (df['annual change'] == max_annual_change)]

# Extract the location
city = outlier_row['location'].values[0]

print(f"Final Answer: {city}")