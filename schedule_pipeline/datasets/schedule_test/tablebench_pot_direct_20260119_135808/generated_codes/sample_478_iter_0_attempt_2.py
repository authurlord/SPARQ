import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'total passengers' to integer
df['total passengers'] = df['total passengers'].astype(int)

# Clean 'annual change' by removing '%' and converting to float
df['annual change'] = df['annual change'].str.replace('%', '').astype(float)

# Identify cities with unusually high total passengers (e.g., > 10 million) and high annual change (e.g., > 100%)
high_passengers = df[df['total passengers'] > 10000000]
high_annual_change = df[df['annual change'] > 100]

# Find intersection of both conditions
unusual_city = pd.merge(high_passengers, high_annual_change, on='location')

# Extract the city name
if not unusual_city.empty:
    city = unusual_city['location'].iloc[0]
else:
    city = "None"

print(f"Final Answer: {city}")