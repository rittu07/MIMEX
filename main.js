const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');

function createWindow() {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1024,
        minHeight: 768,
        icon: path.join(__dirname, 'assets/icon.png'), // Ensure you have an icon
        title: "MIMEX Helrit_bot Control Station",
        backgroundColor: '#0f172a', // Matches our new slate theme
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false, // Simplified for local prototype
            webSecurity: false, // Allow loading local files (CORS/File protocol)
            allowRunningInsecureContent: true
        },
        frame: true, // Standard OS window frame
        autoHideMenuBar: true // Modern look, hide default ugly menu
    });

    // Load the index.html of the app.
    mainWindow.loadFile('index.html');

    // Open the DevTools.
    // mainWindow.webContents.openDevTools();
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.whenReady().then(() => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});
