import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove rows with invalid or inconsistent entries
# Filter out the row where year is '191822' and malaria has '(avg)'
df_clean = df[df['year'] != '191822'].copy()

# Convert typhus and typhoid fever to numeric (some might be strings)
df_clean['typhus'] = pd.to_numeric(df_clean['typhus'], errors='coerce')
df_clean['typhoid fever'] = pd.to_numeric(df_clean['typhoid fever'], errors='coerce')

# Drop any rows with NaN after conversion
df_clean.dropna(subset=['typhus', 'typhoid fever'], inplace=True)

# Compute the correlation between typhus and typhoid fever
correlation = df_clean['typhus'].corr(df_clean['typhoid fever'])

print(f"Final Answer: {correlation:.2f}")