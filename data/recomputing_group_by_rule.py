import pandas as pd

def inefficient_recompute_group_by():
    # Sample DataFrame
    data = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'value': [10, 20, 15, 30, 25, 5]
    })

    category_sums = data.groupby('category').sum()
    print("Category Sums:\n", category_sums)
    
    # Some unrelated processing
    total = data['value'].sum()
    print("Total Value:", total)
    
    # First violation: Recompute mean using the same groupby on 'category'
    category_means = data.groupby('category').mean()
    print("Category Means:\n", category_means)

    # Some unrelated processing
    total = data['value'].sum()
    print("Total Value:", total)

    # Second violation: Recompute mean using the same groupby on 'category'
    category_means = data.groupby('category').mean()
    print("Category Means:\n", category_means)