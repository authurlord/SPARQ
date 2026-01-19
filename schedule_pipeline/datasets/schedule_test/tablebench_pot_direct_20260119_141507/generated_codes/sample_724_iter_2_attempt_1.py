import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'Power output (hp)' column
def clean_power_output(x):
    if pd.isna(x) or x == '-' or x == '':
        return 0
    elif '/' in str(x):
        parts = str(x).split('/')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return 0
    else:
        return float(x)

# Apply cleaning to the column
df['Power output (hp)'] = df['Power output (hp)'].apply(clean_power_output)

# Sort by power output in descending order and get the rank
ranked_models = df[['Model', 'Power output (hp)']].sort_values(by='Power output (hp)', ascending=False)

# Extract model names in ranked order
ranked_model_names = ranked_models['Model'].tolist()

print(f"Final Answer: {', '.join(ranked_model_names)}")