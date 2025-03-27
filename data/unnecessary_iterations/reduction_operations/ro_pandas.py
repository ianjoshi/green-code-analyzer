import pandas as pd

def reduction_operations_pandas():
    """
    This function demonstrates the reduction operation code smell in Pandas:
    1. Using loops to compute sums instead of df['column'].sum().
    2. Using loops to compute means instead of df['column'].mean().
    3. Using loops to compute min and max instead of df['column'].min() and df['column'].max().
    """
    # Create a sample DataFrame
    df = pd.DataFrame({'values': pd.np.random.rand(100)})
    
    # Violation 1: Element-wise sum instead of df['values'].sum()
    total = 0
    for i in range(len(df)):
        total += df['values'][i]
    
    # Violation 2: Element-wise mean instead of df['values'].mean()
    total_mean = 0
    for i in range(len(df)):
        total_mean += df['values'][i]
    mean = total_mean / len(df)
    
    # Violation 3: Element-wise min instead of df['values'].min()
    min_value = df['values'][0]
    for i in range(1, len(df)):
        if df['values'][i] < min_value:
            min_value = df['values'][i]
    
    # Violation 4: Element-wise max instead of df['values'].max()
    max_value = df['values'][0]
    for i in range(1, len(df)):
        if df['values'][i] > max_value:
            max_value = df['values'][i]
    
    return total, mean, min_value, max_value
