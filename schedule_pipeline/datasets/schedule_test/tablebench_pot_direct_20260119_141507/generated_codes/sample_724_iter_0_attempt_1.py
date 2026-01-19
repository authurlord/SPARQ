import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract power output column and clean values
power_output = df['Power output (hp)'].astype(str)

# Replace "200/235" with average (217.5), and convert to numeric
def parse_power(val):
    if val == '200/235':
        return 217.5
    elif val == '300':
        return 300
    else:
        return float(val)

cleaned_power = [parse_power(val) for val in power_output]
df['Power output (hp)'] = cleaned_power

# Sort by power output in descending order and get the model names
ranked_models = df[['Model', 'Power output (hp)']].sort_values(by='Power output (hp)', ascending=False)

# Extract only the model names in order
ranked_list = ranked_models['Model'].tolist()

print(f"Final Answer: {', '.join(ranked_list)}")