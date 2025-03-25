import pandas as pd

def inefficient_chain_indexing(data: dict) -> tuple:
    # Create a sample DataFrame
    df = pd.DataFrame(data)
    
    # First violation of ChainIndexingRule: df['stats']['average']
    avg_value = df['stats']['average']
    
    # Some intermediate processing
    temp = avg_value * 2
    
    # Second violation of ChainIndexingRule: df['metrics']['count']
    count_value = df['metrics']['count']
    
    return (avg_value, count_value)