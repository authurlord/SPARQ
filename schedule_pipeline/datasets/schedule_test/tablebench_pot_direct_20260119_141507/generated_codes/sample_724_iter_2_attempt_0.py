import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean 'Power output (hp)' column: handle '200/235' and '-' 
def clean_power_output(value):
    if pd.isna(value) or value == '-':
        return np.nan
    elif '/' in str(value):
        return max(map(int, str(value).split('/')))
    else:
        return float(value)

# Apply cleaning
df['Power output (hp)'] = df['Power output (hp)'].apply(clean_power_output)

# Sort by power output (hp) in descending order and get the rank
ranked_models = df.sort_values(by='Power output (hp)', ascending=False)[['Model', 'Power output (hp)']]

# Extract model names in order of highest to lowest power output
ranked_model_names = ranked_models['Model'].tolist()

print(f"Final Answer: {', '.join(ranked_model_names)}")