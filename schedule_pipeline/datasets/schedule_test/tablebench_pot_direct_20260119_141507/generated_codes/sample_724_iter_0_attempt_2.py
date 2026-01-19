import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract the 'Power output (hp)' column
power_output = df['Power output (hp)'].astype(str)

# Clean and convert power output to numeric
def parse_power(output):
    if output == '300':
        return 300
    elif output == '200/235':
        return 235
    elif output == '180':
        return 180
    elif output == '150':
        return 150
    elif output == '140':
        return 140
    else:
        return 0

# Apply parsing to each value
power_numeric = df['Model'].apply(lambda x: parse_power(power_output.iloc[df['Model'].index[df['Model'] == x].tolist()[0]]))

# Create a new DataFrame with model and power output
models_power = df[['Model', 'Power output (hp)']].copy()
models_power['Power output (hp)'] = models_power['Power output (hp)'].apply(parse_power)

# Sort by power output in descending order
ranked_models = models_power.sort_values(by='Power output (hp)', ascending=False)

# Extract the ranked list of models
ranked_list = ranked_models['Model'].tolist()

print(f"Final Answer: {', '.join(ranked_list)}")