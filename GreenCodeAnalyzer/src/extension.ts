import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as childProcess from 'child_process';

export function activate(context: vscode.ExtensionContext) {
    console.log('GreenCodeAnalyzer is now active!');

    // Register command for button press
    const disposable = vscode.commands.registerCommand('greencodeanalyzer.runAnalyzer', async () => {
        // Get the active text editor
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active file found.');
            return;
        }

        // Get file path
        const filePath = editor.document.fileName;

        // Read file content
        fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
                vscode.window.showErrorMessage('Error reading the file.');
                return;
            }

            // Run all the analyzer logic 
            runAnalyzer(filePath, data);
        });
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}

function runAnalyzer(filePath: string, fileContent: string) {

    // Get the project root directory using VS Code workspace folder
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('Workspace not found.');
        return;
    }

    // For now this is hardcoded, maybe change it after to be dynamic	
    const projectRoot = path.dirname(workspaceFolder);

    // Define the path to the 'data' folder relative to the parent folder
    const dataFolder = path.join(projectRoot, 'data');
    
    // Ensure 'data' folder exists
    if (!fs.existsSync(dataFolder)) {
        fs.mkdirSync(dataFolder, { recursive: true });
    }

    const fileName = path.basename(filePath);
    const destinationPath = path.join(dataFolder, fileName);

    // Write file content
    fs.writeFile(destinationPath, fileContent, 'utf8', (err) => {
        if (err) {
            vscode.window.showErrorMessage(`Failed to save file: ${err.message}`);
        } else {
            vscode.window.showInformationMessage(`File saved to ${destinationPath}`);
            runMainScript(projectRoot);  // Run `main.py` from the project root
        }
    });
}

function runMainScript(projectRoot: string) {
    const mainScriptPath = path.join(projectRoot, 'main.py');

    if (!fs.existsSync(mainScriptPath)) {
        vscode.window.showErrorMessage(`main.py not found at ${mainScriptPath}`);
        return;
    }

    // Run the script using Python
    const pythonProcess = childProcess.spawn('python', [mainScriptPath], { cwd: projectRoot });

    pythonProcess.stdout.on('data', (data) => {
        vscode.window.showInformationMessage(`Output: ${data.toString()}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        vscode.window.showErrorMessage(`Error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
        vscode.window.showInformationMessage(`main.py exited with code ${code}`);
    });
}
