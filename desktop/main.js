const { app, BrowserWindow } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn, spawnSync } = require("child_process");

let backendProcess = null;

function commandExists(command, args = ["--version"]) {
  const result = spawnSync(command, args, { stdio: "ignore" });
  return !result.error;
}

function startBackend() {
  const backendDir = app.isPackaged
    ? path.join(process.resourcesPath, "backend")
    : path.resolve(__dirname, "..");

  const runtimeBaseDir = path.join(app.getPath("userData"), "runtime");
  const runtimeEnvPath = path.join(runtimeBaseDir, ".env");
  fs.mkdirSync(runtimeBaseDir, { recursive: true });

  const baseArgs = ["-m", "uvicorn", "lol_stat_tracker.api:app", "--host", "127.0.0.1", "--port", "8000"];
  const commandCandidates = process.platform === "win32"
    ? [["python", baseArgs], ["py", ["-3", ...baseArgs]]]
    : [["python3", baseArgs], ["python", baseArgs]];

  const env = {
    ...process.env,
    TRACKER_BASE_DIR: runtimeBaseDir,
    TRACKER_ENV_PATH: runtimeEnvPath
  };

  for (const [command, args] of commandCandidates) {
    if (!commandExists(command, args[0] === "-3" ? ["-3", "--version"] : ["--version"])) {
      continue;
    }
    backendProcess = spawn(command, args, { cwd: backendDir, stdio: "inherit", env });
    return;
  }

  throw new Error("Could not start backend. Install Python 3 and make sure it is on PATH.");
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 920,
    minHeight: 700,
    webPreferences: {
      contextIsolation: true
    }
  });
  window.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
});

