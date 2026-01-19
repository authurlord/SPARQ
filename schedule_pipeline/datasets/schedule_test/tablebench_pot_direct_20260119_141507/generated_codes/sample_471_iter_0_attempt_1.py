import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation, prominence, and col to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')
df['col (m)'] = pd.to_numeric(df['col (m)'], errors='coerce')

# Calculate mean and std for each column
mean_elev = df['elevation (m)'].mean()
std_elev = df['elevation (m)'].std()

mean_prominence = df['prominence (m)'].mean()
std_prominence = df['prominence (m)'].std()

mean_col = df['col (m)'].mean()
std_col = df['col (m)'].std()

# Identify outliers using 2 standard deviations rule
outliers = []
for idx, row in df.iterrows():
    elev = row['elevation (m)']
    prominence = row['prominence (m)']
    col_val = row['col (m)']
    
    if (abs(elev - mean_elev) > 2 * std_elev or 
        abs(prominence - mean_prominence) > 2 * std_prominence or 
        abs(col_val - mean_col) > 2 * std_col):
        outliers.append(row['peak'])

# Remove duplicates and print
unique_outliers = list(set(outliers))
print(f"Final Answer: {', '.join(unique_outliers)}")