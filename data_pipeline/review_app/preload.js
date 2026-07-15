const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  getState: () => ipcRenderer.invoke('get-state'),
  setScore: (entryId, score) => ipcRenderer.invoke('set-score', { entryId, score }),
  setGt: (entryId, fields) => ipcRenderer.invoke('set-gt', { entryId, fields }),
  pickDataFile: () => ipcRenderer.invoke('pick-data-file'),
});
