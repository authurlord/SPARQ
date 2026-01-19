import pandas as pd

df = pd.read_csv('table.csv')
# Extract the values for Conservative councillors in 1947 and 1972
initial_value = df[df['Year'] == '1947']['Conservative councillors'].values[0]
final_value = df[df['Year'] == '1972']['Conservative councillors'].values[0]

# Calculate the average annual change
years_span = 1972 - 1947
average_change = (int(final_value) - int(initial_value)) / years_span

print(f"Final Answer: {average_change:.2f}")