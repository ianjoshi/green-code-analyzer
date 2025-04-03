# Contributing to GreenCodeAnalyzer

Thank you for your interest in contributing to **GreenCodeAnalyzer**! We welcome contributions that help improve the tool and make it more effective for identifying energy-inefficient patterns in Python code.

## How to Contribute

### Reporting Issues
If you encounter a bug or have a feature request, please:
1. Check the [issue tracker](https://github.com/ianjoshi/sustainablese-g1-green-shift-left/issues) to see if it has already been reported.
2. If not, create a new issue with:
   - A clear and descriptive title.
   - Steps to reproduce the issue (if applicable).
   - Any relevant logs, screenshots, or code snippets.

### Submitting Code Changes
We accept contributions in the form of bug fixes, new features, or documentation updates. Follow these steps to contribute:

1. **Fork the Repository**  
   Fork the repository to your GitHub account and clone it locally:
   ```bash
   git clone https://github.com/ianjoshi/green-code-analyzer.git
   cd green-code-analyzer
   ```

2. **Set Up the Development Environment**  
   Install the required dependencies:
   ```bash
   # Using conda
   conda env create -f environment.yml
   conda activate gsl_venv
   
   # Or using pip
   pip install -r requirements.txt
   npm install
   npm run compile
   ```

3. **Make Your Changes**  
   - Follow the existing code style and structure as described [here](#code-style-guidelines).
   - Add new rules as described [here](#adding-new-rules).	
   - Add or update files in the `GreenCodeAnalyzer/data/tests` directory to cover your changes.
   - Run tests through main.py to ensure your changes add the desired functionality. 
      - Change the file path in `main.py` to the file you want to test.
      - Run the following command to execute the tests:
      ```bash
      python main.py
      ```

4. **Commit Your Changes**  
   Write clear and concise commit messages:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

5. **Push and Create a Pull Request**  
   Push your changes to your fork and create a pull request:
   ```bash
   git push origin <your-branch-name>
   ```
   - Go to the original repository and open a pull request.
   - Provide a detailed description of your changes and link any related issues.

### Code Style Guidelines
- Use **TypeScript** for the VS Code extension and **Python** for the analysis rules.

### Adding New Rules
If you want to add a new energy-efficiency rule:
1. Create a new file in the `GreenCodeAnalyzer/rules` directory.
2. Implement the rule by extending the `BaseRule` class.
3. Add a test file for the rule in the `GreenCodeAnalyzer/data/tests` directory.
4. Update the `README.md` to document the new rule under the **Supported Rules** section.

### Running the Extension Locally
To test the extension:
1. Open the project in VS Code.
2. Press `F5` or run the command **Debug: Start Debugging** from the Command Palette (`Ctrl+Shift+P`).
3. Use the new VS Code window to test your changes.

### Documentation Updates
If you improve or add documentation:
- Update the `README.md` or create new markdown files as needed.
- Ensure the documentation is clear and concise.

## Code of Conduct
By contributing, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?
If you have any questions, feel free to open an issue or reach out to the maintainers.

Thank you for contributing to GreenCodeAnalyzer!