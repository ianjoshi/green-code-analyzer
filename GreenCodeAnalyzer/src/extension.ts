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

// Add a variable to store active decoration types
let activeDecorationTypes: vscode.TextEditorDecorationType[] = [];

// Track processed lines to avoid duplicate hover messages
interface LineMessageInfo {
  messages: vscode.MarkdownString[];
  nutriScore: string;
}

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

      // Check if the file exists
      if (!fs.existsSync(filePath)) {
        vscode.window.showErrorMessage(`File not found: ${filePath}`);
        return;
      }

      // Show message that analysis is starting
      const progressMessage = vscode.window.showInformationMessage(
        "GreenCodeAnalyzer is analyzing your code...",
        { modal: false }
      );

      // Use extension path instead of workspace folder
      // This ensures the extension works in any workspace
      const extensionPath = context.extensionPath;
      
      // Run the main script directly with the file path
      runMainScript(extensionPath, filePath, context, progressMessage);
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
}

// Run the main script from the extension directory to analyze the file
function runMainScript(extensionPath: string, filePath: string, context: vscode.ExtensionContext, progressMessage: Thenable<any>) {
  // Look for main.py in the extension's src directory
  const mainScriptPath = path.join(extensionPath, "main.py");

  if (!fs.existsSync(mainScriptPath)) {
    vscode.window.showErrorMessage(`main.py not found at ${mainScriptPath}`);
    return;
  }

  // Run the Python process with the extension directory as CWD
  const pythonProcess = childProcess.spawn("python", [mainScriptPath, filePath], {
    cwd: path.dirname(mainScriptPath), // Use the src directory as working directory
  });

  let stdoutData = "";
  pythonProcess.stdout.on("data", (data) => {
    stdoutData += data.toString();
  });

  // Process all output when the process ends
  pythonProcess.on("close", () => {
    // Dismiss the progress message
    progressMessage.then(undefined, undefined);
    
    if (stdoutData) {
      processAnalyzerOutput(stdoutData, context);
      // Show analysis completion message
      vscode.window.showInformationMessage("GreenCodeAnalyzer analysis complete!");
    } else {
      vscode.window.showInformationMessage("GreenCodeAnalyzer analysis complete with no issues found.");
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    // Dismiss the progress message
    progressMessage.then(undefined, undefined);
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

  // Regular expression to match the output from the analyzer
  const regex =
    /Rule ID: (.+?), Rule Name: (.+?), Description: (.+?)(?:, Penalty: (.+?))?, Optimization: (.+?), Affected Line\(s\): (?:Line (\d+)|Lines (\d+)-(\d+))/g;
  
  // Keep track of line info to avoid duplicate hover messages
  const lineInfoMap: Map<number, LineMessageInfo> = new Map();

  // Clear previous decorations
  clearDecorations();

  let match;
  while ((match = regex.exec(output)) !== null) {
    const ruleName = match[2];
    const description = match[3];
    const penalty = parseFloat(match[4]);
    const optimization = match[5];
    
    // Get the line number(s) - handle both single line and range formats
    let startLine, endLine;
    if (match[6]) {
      startLine = parseInt(match[6]) - 1;
      endLine = startLine;
    } else {
      startLine = parseInt(match[7]) - 1;
      endLine = parseInt(match[8]) - 1;
    }
    
    const nutriScore = getNutriScore(penalty);

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

    // Apply decorations to all lines in the range
    for (let lineNumber = startLine; lineNumber <= endLine; lineNumber++) {
      try {
        // Make sure the line exists in the document
        if (lineNumber >= 0 && lineNumber < editor.document.lineCount) {
          // Get or create line info
          if (!lineInfoMap.has(lineNumber)) {
            lineInfoMap.set(lineNumber, {
              messages: [hoverMessage],
              nutriScore: nutriScore
            });
          } else {
            // Check if this exact message is already in the array to avoid duplicates
            const lineInfo = lineInfoMap.get(lineNumber)!;
            const isDuplicate = lineInfo.messages.some(msg => 
              msg.value === hoverMessage.value
            );
            
            if (!isDuplicate) {
              lineInfo.messages.push(hoverMessage);
            }
          }
        }
      } catch (error) {
        console.error(`Error processing line ${lineNumber}:`, error);
      }
    }
  }

  // Create decoration types dynamically for each NutriScore level
  const decorationTypes: { [key: string]: vscode.TextEditorDecorationType } = {};
  for (const [score, color] of Object.entries(nutriScoreColors)) {
    decorationTypes[score] = vscode.window.createTextEditorDecorationType({
      isWholeLine: false,
      gutterIconSize: 'contain',
      borderColor: color,
      borderWidth: '0 0 0 3px',
      borderStyle: 'solid',
      overviewRulerColor: color,
      overviewRulerLane: vscode.OverviewRulerLane.Right,
    });

    // Store the decoration type in our active decorations array
    activeDecorationTypes.push(decorationTypes[score]);
  }

  // Process line info and apply decorations by NutriScore
  const decorationsMap: { [key: string]: vscode.DecorationOptions[] } = {
    A: [], B: [], C: [], D: [], E: [], NaN: []
  };

  // Convert the line info map to decoration options
  lineInfoMap.forEach((info, lineNumber) => {
    try {
      const line = editor.document.lineAt(lineNumber);
      const range = new vscode.Range(
        lineNumber, 0, 
        lineNumber, line.text.length
      );

      // If we have multiple messages, combine them
      let combinedMessage: vscode.MarkdownString;
      if (info.messages.length === 1) {
        combinedMessage = info.messages[0];
      } else {
        combinedMessage = new vscode.MarkdownString();
        combinedMessage.appendMarkdown(`## GreenCodeAnalyzer\n`);
        combinedMessage.appendMarkdown(`---\n`);
        
        // Add messages with separators between them
        for (let i = 0; i < info.messages.length; i++) {
          const msgContent = info.messages[i].value;
          // Strip the header from all but the first message
          const contentWithoutHeader = msgContent.replace(/^## GreenCodeAnalyzer\n---\n/, '');
          combinedMessage.appendMarkdown(contentWithoutHeader);
          
          // Add separator between messages (except after the last one)
          if (i < info.messages.length - 1) {
            combinedMessage.appendMarkdown(`\n\n---\n\n`);
          }
        }
      }

      const decoration: vscode.DecorationOptions = {
        range: range,
        hoverMessage: combinedMessage
      };

      // Add to decorations map based on severity
      decorationsMap[info.nutriScore].push(decoration);
    } catch (error) {
      console.error(`Error creating decoration for line ${lineNumber}:`, error);
    }
  });

  // Apply all decorations
  for (const [score, decorations] of Object.entries(decorationsMap)) {
    if (decorations.length > 0) {
      editor.setDecorations(decorationTypes[score], decorations);
    }
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
