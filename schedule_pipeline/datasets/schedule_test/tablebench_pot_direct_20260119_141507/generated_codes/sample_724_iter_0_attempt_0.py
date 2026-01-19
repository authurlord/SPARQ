import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'Power output (hp)' column: convert '200/235' to 235, '300' to 300, etc.
def parse_power_output(value):
    if isinstance(value, str):
        if '/' in value:
            return max(map(int, value.split('/')))
        else:
            return int(value)
    return value

df['Power output (hp)'] = df['Power output (hp)'].apply(parse_power_output)

# Sort by power output in descending order and get the model names
ranked_models = df[['Model', 'Power output (hp)']].sort_values(by='Power output (hp)', ascending=False)

# Extract only the model names in order
ranked_list = ranked_models['Model'].tolist()

print(f"Final Answer: {', '.join(ranked_list)}")