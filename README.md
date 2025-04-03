### CS4575 Project 2 - Group 1
# GreenCodeAnalyzer: A VSCode Extension for Static Energy Analysis

**GreenCodeAnalyzer** is a static code analysis tool that identifies energy-inefficient patterns in Python code and suggests optimizations to improve energy consumption. By shifting energy efficiency concerns "left" (earlier) in the development process, developers can make more sustainable coding decisions from the start.

## Features

- **Static Energy Analysis**: Analyzes Python code without executing it to detect potential energy hotspots
- **Visual Code Annotations**: VS Code extension that provides visual feedback with highlighted energy smells
- **Optimization Suggestions**: Provides specific recommendations to make code more energy-efficient
- **Multiple Rule Detection**: Covers various energy-inefficient patterns common in data science and ML code

### Supported Rules

The extension currently detects these energy code smells:

| Rule                               | Description                                                     | Impact                                                 |
| ---------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------ |
| `batch_matrix_multiplication`      | Sequential matrix multiplications instead of batched operations | Missed hardware acceleration opportunities             |
| `blocking_data_loader`            | Prevent using data loading strategies that stall GPU execution   | The GPU may remain idle while waiting for data             |
| `broadcasting`                     | Inefficient tensor operations that could use broadcasting       | Unnecessary memory allocations                         |
| `calculating_gradients`            | Computing gradients when not needed for training                | Unnecessary computation overhead                       |
| `chain_indexing`                   | Chained Pandas DataFrame indexing operations                    | Extra intermediate objects creation                    |
| `conditional_operations`           | Element-wise conditional operations in loops                    | Inefficient branching and repeated calculations        |
| `data_parallelization`          | Usage of torch.nn.DataParallel detected                            | Less efficient than DistributedDataParallel               |
| `element_wise_operations`          | Element-wise operations in loops                                | Inefficient iteration instead of vectorized operations |
| `excessive_gpu_transfers`          | Frequent CPU-GPU tensor transfers                               | High data movement overhead                            |
| `excessive_training`               | Training loops without early stopping mechanisms                | Wasted computation after model convergence             |
| `filter_operations`                | Manual filtering in loops instead of vectorized operations      | Increased CPU workload                                 |
| `ignoring_inplace_ops`             | Operations that could use in-place variants                     | Unnecessary memory allocations                         |
| `inefficient_caching_of_common_arrays`        | Recreating identical arrays inside loops                        | Redundant memory and CPU usage                         |
| `inefficient_data_loader_transfer` | DataLoader transfer to GPU without pin_memory                  | Increased data transfer time                           |
| `inefficient_df_joins`             | Repeated merges or merges without DataFrame indexing            | High memory usage and increased computation time       |
| `inefficient_iterrows`             | Inefficient row-by-row Pandas iterations                        | Python overhead for operations                         |
| `large_batch_size_causing_memory_swapping` | Batch sizes causing memory swapping                             | Excessive disk I/O and system slowdown                 |
| `recomputing_group_by`             | Repetitive group by operations on the same data                 | Redundant computation and memory usage                 |
| `reduction_operations`             | Manual reduction operations using loops                         | Missed vectorization opportunities                     |
| `redundant_model_refitting`        | Redundant retraining of models with unchanged data              | Wasteful recalculation                                 |

## Installing the Extension from VS Code Marketplace

You can install the GreenCodeAnalyzer extension directly from the **VS Code Marketplace**:

1. Open VS Code
2. Go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window or press `Ctrl+Shift+X`
3. Search for "GreenCodeAnalyzer" and click on the install button.

Alternatively, you can install it from the [VS Code Marketplace website](https://marketplace.visualstudio.com/items?itemName=KevinHoxha.GreenCodeAnalyzer).

### Using the Extension

Once installed, you can analyze your Python code for energy inefficiencies:

1. Open a Python file in VS Code
2. Use one of the following methods to run the analyzer:
   - Press `Ctrl+Shift+P` to open the Command Palette, then type and select "GreenCodeAnalyzer: Run Analyzer"
   - Right-click in the editor and select "Run GreenCodeAnalyzer" from the context menu

The analysis results will appear as decorations in your code editor, highlighting potential energy inefficiencies with suggestions for improvement.

To clear the analysis markers:

- Press `Ctrl+Shift+P` and select "GreenCodeAnalyzer: Clear Gutters"
- Or right-click and select "Clear GreenCodeAnalyzer Gutters"

### Interpreting Results

Each detected code smell includes:

- A description of the energy inefficiency
- A specific recommendation for optimization

## Running the Extension Locally

### Requirements

Python 3.10 or higher
- VS Code
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - torch
  - tensorflow

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ianjoshi/sustainablese-g1-green-shift-left.git
   cd sustainablese-g1-green-shift-left
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

3. Open the file you are working on in the new VS Code window that opens. 

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
