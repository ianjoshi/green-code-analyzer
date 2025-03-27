import sys
from engines.smell_engine import SmellEngine

# Example
if __name__ == "__main__":
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/test.py"
        
    collector = SmellEngine(file_path)
    smells_dict = collector.collect()
    
    print("Detected Code Smells:\n" + "=" * 30)
    for line, smells in smells_dict.items():
        print(f"\nLine {line}:")
        for smell in smells:
            print(f"  - {smell}")
