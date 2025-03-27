import pandas as pd

def process_dataframe(data: dict) -> tuple:
    # Create a sample DataFrame
    df = pd.DataFrame(data)
    
    # First violation of ChainIndexingRule: df['stats']['average']
    avg_value = df['stats']['average']
    
    # Some intermediate processing
    temp = avg_value * 2
    
    # Second violation of ChainIndexingRule: df['metrics']['count']
    count_value = df['metrics']['count']
    
    return (avg_value, count_value)

# Example usage
data = {
    'stats': {'average': 15.5, 'max': 30.0},
    'metrics': {'count': 100, 'sum': 1550.0}
}
result = process_dataframe(data)
print(result)  # Outputs: (15.5, 100)