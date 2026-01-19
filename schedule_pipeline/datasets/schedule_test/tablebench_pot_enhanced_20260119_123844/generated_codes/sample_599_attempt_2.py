import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 1944
df_1944 = df[df['Year'] == '1944']
# Extract US Chart position and convert to numeric, coercing errors to NaN
df_1944['US Chart position'] = pd.to_numeric(df_1944['US Chart position'], errors='coerce')
# Calculate average US Chart position
avg_chart_position = df_1944['US Chart position'].mean()
print(f"Final Answer: {avg_chart_position:.1f}")