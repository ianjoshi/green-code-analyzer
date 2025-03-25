from engines.smell_engine import SmellEngine

# Example
if __name__ == "__main__":
    collector = SmellEngine("data/long_loop.py")
    smells_dict = collector.collect()
    
    print("Detected Code Smells:\n" + "=" * 30)
    for line, smells in smells_dict.items():
        print(f"\nLine {line}:")
        for smell in smells:
            print(f"  - {smell}")
