from engines.smell_engine import SmellEngine

# Example
if __name__ == "__main__":
    collector = SmellEngine("data/test.py")
    smells = collector.collect()
    
    for smell in smells:
        print(smell)