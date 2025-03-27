import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def process_data_inefficiently(customers, orders, products, order_items):
    """
    Process data using inefficient join operations.
    This function demonstrates multiple violations of efficient DataFrame join practices.
    """
    # Violation 1: Joining large DataFrames without setting index first
    # This causes inefficient lookups during the join operation
    sales_data = orders.merge(customers, on='customer_id', how='left')
    
    # Some intermediate operations
    sales_data['order_year'] = sales_data['order_date'].dt.year
    
    # Violation 2: Redundant joins that could be avoided
    # Joining the same DataFrames multiple times for different operations
    electronics_sales = orders.merge(customers, on='customer_id', how='left')
    electronics_sales = electronics_sales.merge(order_items, on='order_id', how='left')
    electronics_sales = electronics_sales.merge(products, on='product_id', how='left')
    electronics_sales = electronics_sales[electronics_sales['category'] == 'Electronics']
    
    clothing_sales = orders.merge(customers, on='customer_id', how='left')
    clothing_sales = clothing_sales.merge(order_items, on='order_id', how='left')
    clothing_sales = clothing_sales.merge(products, on='product_id', how='left')
    clothing_sales = clothing_sales[clothing_sales['category'] == 'Clothing']
    
    return {
        'electronics_sales': electronics_sales,
        'clothing_sales': clothing_sales
    }

# Example usage
if __name__ == "__main__":

    # Create a large customer DataFrame
    customers = pd.DataFrame({
        'customer_id': range(1, 100001),
        'name': [f'Customer {i}' for i in range(1, 100001)],
        'age': np.random.randint(18, 80, 100000),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 100000)
    })
    
    # Create a large orders DataFrame
    orders = pd.DataFrame({
        'order_id': range(1, 500001),
        'customer_id': np.random.randint(1, 100001, 500000),
        'order_date': pd.date_range('2020-01-01', periods=500000, freq='h'),
        'amount': np.random.uniform(10, 1000, 500000).round(2)
    })
    
    # Create a products DataFrame
    products = pd.DataFrame({
        'product_id': range(1, 10001),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], 10000),
        'price': np.random.uniform(5, 500, 10000).round(2)
    })
    
    # Create order_items DataFrame
    order_items = pd.DataFrame({
        'item_id': range(1, 1000001),
        'order_id': np.random.randint(1, 500001, 1000000),
        'product_id': np.random.randint(1, 10001, 1000000),
        'quantity': np.random.randint(1, 10, 1000000)
    })
    
    # Process the data inefficiently
    sales_results = process_data_inefficiently(customers, orders, products, order_items)

    # Display the results
    print("Electronics Sales Data:")
    print(sales_results['electronics_sales'].head())
