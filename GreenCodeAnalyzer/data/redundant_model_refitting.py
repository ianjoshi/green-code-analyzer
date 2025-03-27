from sklearn.linear_model import LogisticRegression

def inefficient_redundant_model_refitting():
    # Sample dataset
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 0]
    
    # Create a model instance
    model = LogisticRegression()
    
    # First redundant fit call with the same data
    model.fit(X, y)  # This is the initial fit
    
    # Some unrelated operation
    print("Model trained once")
    
    # Second redundant fit call with identical data
    model.fit(X, y)  # This violates RedundantModelRefittingRule
    
    # Another unrelated operation
    model.predict([[1, 2]])
    
    # Third redundant fit call with the same data
    model.fit(X, y)  # This also violates RedundantModelRefittingRule
    
    return model