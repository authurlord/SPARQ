import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the years 1947 and 1972
start_value = df[df['Year'] == '1947']['Conservative councillors'].values[0]
end_value = df[df['Year'] == '1972']['Conservative councillors'].values[0]

# Calculate average annual change
annual_change = (int(end_value) - int(start_value)) / (1972 - 1947)
print(f"Final Answer: {annual_change:.2f}")