import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Conservative councillors' to integers
conservative_councillors = df['Conservative councillors'].astype(int)
# Get the values for 1947 and 1972
start_value = conservative_councillors.iloc[0]
end_value = conservative_councillors.iloc[-1]
# Calculate average annual change
annual_change = (end_value - start_value) / (1972 - 1947)
print(f"Final Answer: {annual_change:.2f}")