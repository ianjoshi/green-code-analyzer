import pandas

def conditional_operation_pandas():
    """
    This function demonstrates the conditional operation code smell in Pandas.
    """
    # Create a sample DataFrame
    df = pandas.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    
    # Violation: Using loops to conditionally modify elements instead of using vectorized operations
    for i in range(len(df)):
        if df.loc[i, "A"] > 2:
            df.loc[i, "B"] = df.loc[i, "B"] + 1  
        else:
            df.loc[i, "B"] = df.loc[i, "B"] - 1  
    
    return df