import pandas as pd

def filter_operations_pandas():
    """
    This function demonstrates the filer operation code smell in Pandas.
    """
    # Create a sample DataFrame
    df = pd.DataFrame({'values': pd.np.random.rand(100)})
    
    # Violation: Using loops to filter elements instead of using boolean indexing (df[df['values'] > 0.5])
    filtered_elements = []
    for i in range(len(df)):
        if df['values'][i] > 0.5:
            filtered_elements.append(df['values'][i])

    return filtered_elements
