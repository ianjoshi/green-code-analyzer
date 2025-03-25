### CS4575 Project 2 - Group 1
# GreenCodeAnalyzer: A VSCode Extension for Static Energy Analysis

**Green Shift Left** is a static code analysis tool that identifies energy-inefficient patterns in Python code and suggests optimizations to improve energy consumption. By shifting energy efficiency concerns "left" (earlier) in the development process, developers can make more sustainable coding decisions from the start.

## Features

- **Static Energy Analysis**: Analyzes Python code without executing it to detect potential energy hotspots
- **Visual Code Annotations**: VS Code extension that provides visual feedback with highlighted energy smells
- **Optimization Suggestions**: Provides specific recommendations to make code more energy-efficient
- **Multiple Rule Detection**: Covers various energy-inefficient patterns common in data science and ML code

## Supported Rules

The tool currently detects these energy code smells:

| Rule    | Description | Impact |
|---------|-------------|--------|
| `long_loop` | Long-running loops with excessive iterations | High CPU usage over time |
| `batch_matrix_mult` | Sequential matrix multiplications instead of batched operations | Missed hardware acceleration opportunities |
| `broadcasting` | Inefficient tensor operations that could use broadcasting | Unnecessary memory allocations |
| `chain_indexing` | Chained Pandas DataFrame indexing operations | Extra intermediate objects creation |
| `ignoring_inplace_ops` | Operations that could use in-place variants | Unnecessary memory allocations |
| `inefficient_iterrows` | Inefficient row-by-row Pandas iterations | Python overhead for operations |

## Getting Started

### Prerequisites

- Python 3.11 or higher
- VS Code
- Required Python packages:
  - pandas
  - numpy
  - torch
  - tensorflow (optional, for TensorFlow-specific rules)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/green-shift-left.git
   cd green-shift-left
   ```

2. Install required dependencies:
   ```bash
   # Using conda
   conda env create -f environment.yml
   conda activate gsl_venv
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. Compile the VS Code extension (from the extension directory):
   ```bash
   cd GreenCodeAnalyzer
   npm install
   npm run compile
   ```

## Usage

### Command Line Analysis

You can analyze Python files directly from the command line:

```bash
python main.py
```

By default, the tool will analyze the code in the `data/test.py` file. You can also specify a custom file by replacing this path.

Running `main.py` will output detected code smells with their line numbers, descriptions, and suggested optimizations in the terminal.

### VS Code Extension

Alternatively, you can use the VS Code extension for a more interactive experience:

1. Inside the editor, open `GreenCodeAnalyzer/src/extension.ts` and press `F5` or run the command **Debug: Start Debugging** from the Command Palette (`Ctrl+Shift+P`). 
2. When prompted to choose a debug environment, select "VS Code Extension Development".

3. Open a the folder `Test_files` from the repository in the extension and open the file `test.py`.

4. To run the static analysis, run the command **GreenCodeAnalyzer: Run Analyzer** from the Command Palette (`Ctrl+Shift+P`).

5. The tool will analyze your code and display results with:
   - Colored gutter to indicate energy smells
   - Detailed hover information with rule descriptions and optimization suggestions

To clear annotations, run the command **GreenCodeAnalyzer: Clear Gutters** from the Command Palette (`Ctrl+Shift+P`).

## How It Works

The tool works by:
1. Parsing Python code into an Abstract Syntax Tree (AST)
2. Applying predefined rules to identify inefficient patterns
3. Generating appropriate warnings and suggestions
4. Visualizing results in the editor (VS Code extension) or terminal output
