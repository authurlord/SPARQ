import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract the 'Power output (hp)' column
power_output = df['Power output (hp)']

# Clean the values: replace '200/235' with 235 (higher value), keep '300' as 300
cleaned_power = []
for val in power_output:
    if val == '200/235':
        cleaned_power.append(235)
    elif val == '300':
        cleaned_power.append(300)
    else:
        cleaned_power.append(float(val))

# Create a new DataFrame with model and cleaned power
df_power = df[['Model']].copy()
df_power['Power (hp)'] = cleaned_power

# Sort by power output in descending order
df_power_sorted = df_power.sort_values(by='Power (hp)', ascending=False)

# Output the ranked list of models
ranked_models = df_power_sorted['Model'].tolist()

print(f"Final Answer: {', '.join(ranked_models)}")