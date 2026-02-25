const { app, BrowserWindow, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn, spawnSync } = require("child_process");

let backendProcess = null;

function commandExists(command, args = ["--version"]) {
  const result = spawnSync(command, args, { stdio: "ignore" });
  return !result.error;
}

function runBlocking(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: "pipe",
    encoding: "utf8",
    ...options
  });
  if (result.error || result.status !== 0) {
    const details = result.stderr || result.stdout || (result.error && result.error.message) || "Unknown error";
    throw new Error(details.trim());
  }
}

function startBackend() {
  const backendDir = app.isPackaged
    ? path.join(process.resourcesPath, "backend")
    : path.resolve(__dirname, "..");

  const runtimeBaseDir = path.join(app.getPath("userData"), "runtime");
  const runtimeEnvPath = path.join(runtimeBaseDir, ".env");
  const runtimeVenvDir = path.join(runtimeBaseDir, ".venv");
  const runtimeVenvPython = process.platform === "win32"
    ? path.join(runtimeVenvDir, "Scripts", "python.exe")
    : path.join(runtimeVenvDir, "bin", "python");
  const requirementsPath = path.join(backendDir, "requirements.txt");
  const requirementsMtime = fs.existsSync(requirementsPath)
    ? String(fs.statSync(requirementsPath).mtimeMs)
    : "missing";
  const depsStampPath = path.join(runtimeBaseDir, ".deps-stamp");

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
    const versionArgs = args[0] === "-3" ? ["-3", "--version"] : ["--version"];
    if (!commandExists(command, versionArgs)) {
      continue;
    }

    if (!fs.existsSync(runtimeVenvPython)) {
      runBlocking(command, [...args.slice(0, args.length - baseArgs.length), "-m", "venv", runtimeVenvDir], {
        cwd: backendDir,
        env
      });
    }

    const installedStamp = fs.existsSync(depsStampPath)
      ? fs.readFileSync(depsStampPath, "utf8").trim()
      : "";
    if (installedStamp !== requirementsMtime) {
      runBlocking(runtimeVenvPython, ["-m", "pip", "install", "--upgrade", "pip"], {
        cwd: backendDir,
        env
      });
      runBlocking(runtimeVenvPython, ["-m", "pip", "install", "-r", requirementsPath], {
        cwd: backendDir,
        env
      });
      fs.writeFileSync(depsStampPath, requirementsMtime, "utf8");
    }

    backendProcess = spawn(runtimeVenvPython, baseArgs, { cwd: backendDir, stdio: "inherit", env });
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
  try {
    startBackend();
    createWindow();
  } catch (err) {
    dialog.showErrorBox(
      "Backend startup failed",
      `Could not start the backend. Install Python 3.11+ and try again.\n\n${err.message}`
    );
    app.quit();
    return;
  }
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

