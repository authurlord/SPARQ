import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average of 'avg finish' from 1985 to 2004
forecast_avg_finish = df['avg finish'].mean()
print(f"Final Answer: {forecast_avg_finish:.1f}")