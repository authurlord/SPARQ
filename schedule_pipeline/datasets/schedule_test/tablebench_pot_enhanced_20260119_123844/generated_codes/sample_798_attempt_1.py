import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for 1947 and 1972
start_councillors = df[df['Year'] == '1947']['Conservative councillors'].values[0]
end_councillors = df[df['Year'] == '1972']['Conservative councillors'].values[0]
# Calculate average annual change
annual_change = (int(end_councillors) - int(start_councillors)) / (1972 - 1947)
print(f"Final Answer: {annual_change:.2f}")