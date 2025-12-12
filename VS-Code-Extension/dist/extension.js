"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const SidebarProvider_1 = require("./SidebarProvider");
function activate(context) {
    console.log('[Fedrag Extension] ========== ACTIVATE CALLED ==========');
    console.log('[Fedrag Extension] ViewType:', SidebarProvider_1.SidebarProvider.viewType);
    console.log('[Fedrag Extension] Expected view ID from package.json: fedragSidebar');
    console.log('[Fedrag Extension] Extension URI:', context.extensionUri.toString());
    const provider = new SidebarProvider_1.SidebarProvider(context.extensionUri);
    console.log('[Fedrag Extension] SidebarProvider created');
    // Register provider synchronously and immediately - MUST happen before VS Code tries to resolve
    const viewTypeToRegister = SidebarProvider_1.SidebarProvider.viewType;
    console.log('[Fedrag Extension] Registering provider for viewType:', viewTypeToRegister);
    console.log('[Fedrag Extension] ViewType matches expected ID?', viewTypeToRegister === 'fedragSidebar');
    try {
        const registration = vscode.window.registerWebviewViewProvider(viewTypeToRegister, provider, {
            webviewOptions: {
                retainContextWhenHidden: true
            }
        });
        console.log('[Fedrag Extension] ✅ registerWebviewViewProvider SUCCESS');
        console.log('[Fedrag Extension] Registered viewType:', viewTypeToRegister);
        console.log('[Fedrag Extension] Registration Disposable:', registration);
        context.subscriptions.push(registration);
        // Verify registration by trying to get the view
        setTimeout(() => {
            console.log('[Fedrag Extension] Checking if view can be resolved...');
            vscode.commands.executeCommand('workbench.view.extension.fedrag').then(() => {
                console.log('[Fedrag Extension] View command executed successfully');
            }, (err) => {
                console.error('[Fedrag Extension] Error executing view command:', err);
            });
        }, 100);
    }
    catch (error) {
        console.error('[Fedrag Extension] ERROR registering provider:', error);
        console.error('[Fedrag Extension] Error stack:', error.stack);
    }
    context.subscriptions.push(vscode.commands.registerCommand('fedrag.start', async () => {
        console.log('[Fedrag Extension] Command fedrag.start executed');
        // Reveal the Fedrag view container so users can see the chat.
        await vscode.commands.executeCommand('workbench.view.extension.fedrag');
    }));
    console.log('[Fedrag Extension] Activation complete');
}
function deactivate() {
    // Nothing to clean up explicitly.
}
