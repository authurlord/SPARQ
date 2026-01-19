import pandas as pd

df = pd.read_csv('table.csv')

# Check for negative land area (invalid)
negative_area = df[df['land area (km square)'] < 0]

# Check for extreme percentage changes (>15% or <-10%)
extreme_change = df[(df['% change'].str.replace('%', '').astype(float) > 15) | (df['% change'].str.replace('%', '').astype(float) < -10)]

# Check for extremely high or low population values
population_values = df['2011 census'].astype(int)
population_outliers = df[(population_values > population_values.quantile(0.95)) | (population_values < population_values.quantile(0.05))]

# Combine all anomalies
anomalies = pd.concat([negative_area, extreme_change, population_outliers], ignore_index=True)

# Print the names of the regions with anomalies
anomaly_names = anomalies['name'].tolist()
print(f"Final Answer: {', '.join(anomaly_names)}")