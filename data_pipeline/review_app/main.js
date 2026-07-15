const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');

// Default to GT review file; can be switched via Open file…
let dataPath = path.join(__dirname, '..', 'gt_v15_gpt5_samples.jsonl');
let entries = [];

function loadEntries(p) {
  const text = fs.readFileSync(p, 'utf-8');
  entries = text.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
}

function saveEntries() {
  const tmp = dataPath + '.tmp';
  const text = entries.map(e => JSON.stringify(e)).join('\n') + '\n';
  fs.writeFileSync(tmp, text, 'utf-8');
  fs.renameSync(tmp, dataPath);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1900,
    height: 1100,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile('index.html');
}

app.whenReady().then(() => {
  try {
    loadEntries(dataPath);
  } catch (e) {
    console.error('Failed to load default path:', e.message);
  }

  ipcMain.handle('get-state', () => ({ entries, dataPath }));

  ipcMain.handle('set-score', (_evt, { entryId, score }) => {
    const e = entries.find(x => x.entry_id === entryId);
    if (!e) return false;
    e.score = score;
    saveEntries();
    return true;
  });

  ipcMain.handle('set-gt', (_evt, { entryId, fields }) => {
    const e = entries.find(x => x.entry_id === entryId);
    if (!e) return false;
    if ('gt_compressed' in fields) e.gt_compressed = fields.gt_compressed;
    if ('gt_action' in fields) e.gt_action = fields.gt_action;
    if ('gt_rationale' in fields) e.gt_rationale = fields.gt_rationale;
    if ('gt_approved' in fields) e.gt_approved = fields.gt_approved;
    // Recompute derived stats
    const compChars = e.gt_compressed ? e.gt_compressed.length : 0;
    e.gt_chars = compChars;
    e.gt_ratio = e.seg_original_chars
      ? Math.round((compChars / e.seg_original_chars) * 1000) / 1000
      : 0;
    saveEntries();
    return true;
  });

  ipcMain.handle('pick-data-file', async () => {
    const res = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: 'JSONL', extensions: ['jsonl'] }],
    });
    if (res.canceled) return null;
    dataPath = res.filePaths[0];
    loadEntries(dataPath);
    return { entries, dataPath };
  });

  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
