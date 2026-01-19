import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter top 5 ranked countries (rank 1 to 5)
top_5 = df[df['rank'].astype(float).between(1, 5)]

# Extract 2010 and 2011 values as integers
values_2010 = top_5['2010'].astype(int)
values_2011 = top_5['2011'].astype(int)

# Compute growth rates
growth_2010 = ((values_2011 - values_2010) / values_2010) * 100
growth_2011 = ((values_2011 - values_2010) / values_2010) * 100  # Same as above? Wait — this is wrong.

Wait — correction: The growth rate for a year is based on the previous year. But here, we have only 2010 and 2011.  
We need to compute the growth rate from 2010 to 2011 and from 2009 to 2010.

So:
- Growth rate from 2009 to 2010: (2010 - 2009) / 2009
- Growth rate from 2010 to 2011: (2011 - 2010) / 2010

Then, compute the percentage difference between these two growth rates.

Let me fix that.

Step 4: Extract 2009, 2010, 2011 values.
Step 5: Compute growth rate from 2009→2010 and 2010→2011.
Step 6: Take the difference between the two growth rates (in percentage).