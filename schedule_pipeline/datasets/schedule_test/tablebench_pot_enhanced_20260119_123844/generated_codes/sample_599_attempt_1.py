import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 1944
df_1944 = df[df['Year'] == '1944']
# Clean 'US Chart position' by extracting numeric part
df_1944['US Chart position'] = pd.to_numeric(df_1944['US Chart position'].str.replace(r'\(.*\)', '', regex=True), errors='coerce')
# Calculate average US Chart position
avg_position = df_1944['US Chart position'].mean()
print(f"Final Answer: {avg_position:.1f}")