# GreenCodeAnalyzer

**GreenCodeAnalyzer** is a static code analysis tool that identifies energy-inefficient patterns in Python code and suggests optimizations to improve energy consumption. By shifting energy efficiency concerns "left" (earlier) in the development process, developers can make more sustainable coding decisions from the start.

## Features

- **Static Energy Analysis**: Analyzes Python code without execution to detect potential energy hotspots
- **Visual Code Annotations**: Provides visual feedback with highlighted energy smells in the editor
- **Optimization Suggestions**: Offers specific recommendations to make your code more energy-efficient
- **Multiple Rule Detection**: Covers various energy-inefficient patterns common in data science and ML code

### Supported Rules

The extension currently detects these energy code smells:

| Rule                               | Description                                                     | Impact                                                 |
| ---------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------ |
| `batch_matrix_mult`                | Sequential matrix multiplications instead of batched operations | Missed hardware acceleration opportunities             |
| `broadcasting`                     | Inefficient tensor operations that could use broadcasting       | Unnecessary memory allocations                         |
| `calculating_gradients`            | Computing gradients when not needed for training                | Unnecessary computation overhead                       |
| `chain_indexing`                   | Chained Pandas DataFrame indexing operations                    | Extra intermediate objects creation                    |
| `conditional_operations`           | Element-wise conditional operations in loops                    | Inefficient branching and repeated calculations        |
| `element_wise_operations`          | Element-wise operations in loops                                | Inefficient iteration instead of vectorized operations |
| `excessive_gpu_transfers`          | Frequent CPU-GPU tensor transfers                               | High data movement overhead                            |
| `excessive_training`               | Training loops without early stopping mechanisms                | Wasted computation after model convergence             |
| `filter_operations`                | Manual filtering in loops instead of vectorized operations      | Increased CPU workload                                 |
| `ignoring_inplace_ops`             | Operations that could use in-place variants                     | Unnecessary memory allocations                         |
| `ineffective_array_caching`        | Recreating identical arrays inside loops                        | Redundant memory and CPU usage                         |
| `inefficient_df_joins`             | Repeated merges or merges without DataFrame indexing            | High memory usage and increased computation time       |
| `inefficient_iterrows`             | Inefficient row-by-row Pandas iterations                        | Python overhead for operations                         |
| `large_batch_size_memory_swapping` | Batch sizes causing memory swapping                             | Excessive disk I/O and system slowdown                 |
| `long_loop`                        | Long-running loops with excessive iterations                    | High CPU usage over time                               |
| `recomputing_group_by`             | Repetitive group by operations on the same data                 | Redundant computation and memory usage                 |
| `reduction_operations`             | Manual reduction operations using loops                         | Missed vectorization opportunities                     |
| `redundant_model_refitting`        | Redundant retraining of models with unchanged data              | Wasteful recalculation                                 |
| `storing_intermediate_results`     | Storing large intermediate tensors or arrays                    | Excessive memory usage and potential swapping          |
| `unnecessary_precision`            | Using higher precision than needed for the task                 | Wasted computation and memory resources                |

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
