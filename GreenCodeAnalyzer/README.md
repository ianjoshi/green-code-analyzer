# GreenCodeAnalyzer

**GreenCodeAnalyzer** is a VS Code extension that performs static analysis on Python code to identify energy-inefficient patterns, helping developers write more sustainable and environmentally friendly code. The tool particularly focuses on inefficiencies in data science and machine learning code.

## Features

- **Static Energy Analysis**: Analyzes Python code without execution to detect potential energy hotspots
- **Visual Code Annotations**: Provides visual feedback with highlighted energy smells in the editor
- **Optimization Suggestions**: Offers specific recommendations to make your code more energy-efficient


### Supported Rules

The extension currently detects these energy code smells:

| Rule    | Description | Impact |
|---------|-------------|--------|
| `long_loop` | Long-running loops with excessive iterations | High CPU usage over time |
| `batch_matrix_mult` | Sequential matrix multiplications instead of batched operations | Missed hardware acceleration opportunities |
| `broadcasting` | Inefficient tensor operations that could use broadcasting | Unnecessary memory allocations |
| `chain_indexing` | Chained Pandas DataFrame indexing operations | Extra intermediate objects creation |
| `ignoring_inplace_ops` | Operations that could use in-place variants | Unnecessary memory allocations |
| `inefficient_iterrows` | Inefficient row-by-row Pandas iterations | Python overhead for operations |
| `inefficient_df_joins` | Repeated merges or merges without DataFrame indexing | High memory usage and increased computation time |
| `excessive_training` | Training loops without early stopping mechanisms | Wasted computation after model convergence |

## Requirements

- Python 3.10 or higher
- VS Code

## Usage

1. Open a Python file in VS Code
2. Run the command **GreenCodeAnalyzer: Run Analyzer** from the Command Palette (`Ctrl+Shift+P`)
3. The tool will analyze your code and display results with:
   - Colored gutter icons indicating energy smells based on severity
   - Detailed hover information with rule descriptions and optimization suggestions

To clear annotations, run the command **GreenCodeAnalyzer: Clear Gutters** from the Command Palette.

## Known Issues

- The extension only works with Python files
- Some rules may produce false positives depending on the context of your code

## Release Notes

### 0.0.1

- Initial release with 8 energy efficiency rules
- Support for Python data science and machine learning code analysis
- Visual indicators for energy code smells in the editor

<!-- --- -->

## How It Works

GreenCodeAnalyzer parses Python code into an Abstract Syntax Tree (AST) and applies predefined rules to identify inefficient patterns. The results are then visualized in the editor with colored indicators and hover information containing optimization suggestions.