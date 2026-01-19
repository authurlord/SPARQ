import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is 1944
df_1944 = df[df['Year'] == '1944']
# Convert 'US Chart position' to numeric, coercing errors to NaN
df_1944['US Chart position'] = pd.to_numeric(df_1944['US Chart position'], errors='coerce')
# Calculate the mean, ignoring NaN values
average_chart_position = df_1944['US Chart position'].mean()
print(f"Final Answer: {average_chart_position:.1f}")