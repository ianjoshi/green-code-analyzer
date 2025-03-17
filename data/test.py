def inefficient_loop():
    # This loop runs for a very large number of iterations
    for i in range(100000):  # This violates the LongLoopRule
        print(i)  # Inefficient operation inside the loop