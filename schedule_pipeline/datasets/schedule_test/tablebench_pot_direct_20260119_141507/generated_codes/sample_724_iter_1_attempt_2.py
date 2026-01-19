import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract power output column
power_output = df['Power output (hp)']

# Clean and convert to numeric
clean_power = []
for value in power_output:
    if pd.isna(value) or value == '-':
        clean_power.append(np.nan)
    elif '/' in str(value):
        parts = str(value).split('/')
        # Take the maximum of the two values
        clean_power.append(max(float(p) for p in parts))
    else:
        clean_power.append(float(value))

# Replace NaN with 0 for ranking purposes (or remove if preferred)
clean_power = [p if not pd.isna(p) else 0 for p in clean_power]

# Create a copy of the original index (Model) and pair with cleaned power
models = df['Model']
ranked_data = list(zip(models, clean_power))

# Sort by power output in descending order
ranked_data.sort(key=lambda x: x[1], reverse=True)

# Extract just the model names in ranked order
ranked_models = [model for model, _ in ranked_data]

print(f"Final Answer: {', '.join(ranked_models)}")