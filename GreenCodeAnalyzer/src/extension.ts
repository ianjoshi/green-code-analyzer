import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as childProcess from "child_process";

// Define NutriScore colors to highlight lines in extension editor
const nutriScoreColors: { [key: string]: string } = {
  A: "rgba(0, 200, 0, 0.5)",
  B: "rgba(180, 230, 0, 0.5)",
  C: "rgba(255, 200, 0, 0.5)",
  D: "rgba(255, 100, 0, 0.5)",
  E: "rgba(255, 0, 0, 0.5)",
  NaN: "rgba(255, 123, 0, 0.6)",
};

let lineMessages: { [key: number]: string } = {};
// Add a variable to store active decoration types
let activeDecorationTypes: vscode.TextEditorDecorationType[] = [];

export function activate(context: vscode.ExtensionContext) {
  console.log("GreenCodeAnalyzer is now active!");

  // Command to run the analyzer
  const analyzerDisposable = vscode.commands.registerCommand(
    "greencodeanalyzer.runAnalyzer",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active file found.");
        return;
      }

      // Clear any existing decorations when analyzer is run
      clearDecorations();

      const filePath = editor.document.fileName;

      fs.readFile(filePath, "utf8", (err, data) => {
        if (err) {
          vscode.window.showErrorMessage("Error reading the file.");
          return;
        }

        // Run the analyzer
        runAnalyzer(filePath, data, context);
      });
    }
  );

  // Command to clear all gutters
  const clearGuttersDisposable = vscode.commands.registerCommand(
    "greencodeanalyzer.clearGutters",
    () => {
      clearDecorations();
      vscode.window.showInformationMessage("GreenCodeAnalyzer gutters cleared.");
    }
  );

  context.subscriptions.push(analyzerDisposable, clearGuttersDisposable);
}

// Function to clear all active decorations
function clearDecorations() {
  activeDecorationTypes.forEach(decorationType => {
    decorationType.dispose();
  });
  activeDecorationTypes = [];
  lineMessages = {};
}

// First, take the file being edited in the extension and save it to the data folder in the project repository
function runAnalyzer(
  filePath: string,
  fileContent: string,
  context: vscode.ExtensionContext
) {
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!workspaceFolder) {
    vscode.window.showErrorMessage("Workspace not found.");
    return;
  }

  const projectRoot = path.dirname(workspaceFolder);
  const dataFolder = path.join(projectRoot, "data");

  if (!fs.existsSync(dataFolder)) {
    fs.mkdirSync(dataFolder, { recursive: true });
  }

  // Save the file to the data folder
  const fileName = path.basename(filePath);
  const destinationPath = path.join(dataFolder, fileName);

  fs.writeFile(destinationPath, fileContent, "utf8", (err) => {
    if (err) {
      vscode.window.showErrorMessage(`Failed to save file: ${err.message}`);
    } else {
      // Run the main script
      runMainScript(projectRoot, context);
    }
  });
}

// Run the main script from the project repository to analyze the file
function runMainScript(projectRoot: string, context: vscode.ExtensionContext) {
  const mainScriptPath = path.join(projectRoot, "main.py");

  if (!fs.existsSync(mainScriptPath)) {
    vscode.window.showErrorMessage(`main.py not found at ${mainScriptPath}`);
    return;
  }

  const pythonProcess = childProcess.spawn("python", [mainScriptPath], {
    cwd: projectRoot,
  });

  // After main.py is run, we will get the output from the energy analyzer.
  // We call the processAnalyzerOutput function to process this output.
  pythonProcess.stdout.on("data", (data) => {
    processAnalyzerOutput(data.toString(), context);
  });

  pythonProcess.stderr.on("data", (data) => {
    vscode.window.showErrorMessage(`Error: ${data.toString()}`);
  });
}

// We want to parse the output from main.py and highlight the lines in the editor based on the NutriScore.
function processAnalyzerOutput(
  output: string,
  context: vscode.ExtensionContext
) {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  // Use regex to parse the output
  const regex =
    /Rule ID: (.+?), Rule Name: (.+?), Description: (.+?)(?:, Penalty: (.+?))?, Optimization: (.+?), Affected Line\(s\): Line (\d+)/g;
  const decorationsMap: { [key: string]: vscode.DecorationOptions[] } = {
    A: [],
    B: [],
    C: [],
    D: [],
    E: [],
    NaN: [],
  };

  // Clear previous decorations and messages
  clearDecorations();

  let match;
  while ((match = regex.exec(output)) !== null) {
    const ruleName = match[2];
    const description = match[3];
    const penalty = parseFloat(match[4]);
    const optimization = match[5];
    const lineNumber = parseInt(match[6]) - 1; // Turn into 0-indexed line number
    const nutriScore = getNutriScore(penalty); // Get NutriScore level based on penalty

    // Create a hover message that includes the rule details
    const hoverMessage = new vscode.MarkdownString();
    hoverMessage.appendMarkdown(`## GreenCodeAnalyzer\n`);
    hoverMessage.appendMarkdown(`---\n`);
    if (nutriScore === "NaN") {
      hoverMessage.appendMarkdown(`### ${ruleName}\n`);
    } else {
      hoverMessage.appendMarkdown(`**${ruleName}** (NutriScore: ${nutriScore})\n\n`);
    }
    hoverMessage.appendMarkdown(`**Description**: ${description}\n\n`);
    if (!isNaN(penalty)) {
      hoverMessage.appendMarkdown(`**Penalty**: ${penalty}\n\n`);
    }
    hoverMessage.appendMarkdown(`**Optimization**: ${optimization}`);

    // Defines how the editor should visually decorate a part of the text
    const decoration: vscode.DecorationOptions = {
      // Extend the range to cover the entire line for hover purposes
      range: new vscode.Range(
        lineNumber, 
        0, 
        lineNumber, 
        editor.document.lineAt(lineNumber).text.length
      ),
      hoverMessage: hoverMessage
    };

    // Add the decoration to the map and create the message to be displayed when this respective line is clicked
    decorationsMap[nutriScore].push(decoration);
    lineMessages[lineNumber] = `${ruleName}: ${description}. Optimization: ${optimization}`;
  }

  // Create decoration types dynamically for each NutriScore level
  const decorationTypes: { [key: string]: vscode.TextEditorDecorationType } =
    {};
  for (const [score, color] of Object.entries(nutriScoreColors)) {
    decorationTypes[score] = vscode.window.createTextEditorDecorationType({
    // Remove the backgroundColor property to avoid highlighting the entire line
    isWholeLine: false, // Change to false to not highlight the whole line
    gutterIconPath: context.asAbsolutePath(path.join('resources', `nutriscore-${score.toLowerCase()}.svg`)),
    gutterIconSize: 'contain',
    // Alternatively, use a colored bar in the margin
    borderColor: color,
    borderWidth: '0 0 0 3px', // Left border only
    borderStyle: 'solid',
    // Keep the overview ruler indicator
    overviewRulerColor: color,
    overviewRulerLane: vscode.OverviewRulerLane.Right,
  });

    // Store the decoration type in our active decorations array
    activeDecorationTypes.push(decorationTypes[score]);

    // Apply decorations to the editor and keep the message to show when the line is clicked
    editor.setDecorations(decorationTypes[score], decorationsMap[score]);
  }
}

// Function to map the penalty of a rule to a NutriScore level.
// This way we know what color to use to highlight the line.
function getNutriScore(penalty: number): string {
  if(isNaN(penalty)) {
    return "NaN";
  }
  if (penalty <= 5) {
    return "A";
  }
  if (penalty <= 10) {
    return "B";
  }
  if (penalty <= 15) {
    return "C";
  }
  if (penalty <= 20) {
    return "D";
  }
  return "E";
}
