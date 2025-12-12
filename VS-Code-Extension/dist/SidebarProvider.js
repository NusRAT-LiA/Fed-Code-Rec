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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SidebarProvider = void 0;
const vscode = __importStar(require("vscode"));
const axios_1 = __importDefault(require("axios"));
class SidebarProvider {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
        console.log('[Fedrag SidebarProvider] Constructor called');
        console.log('[Fedrag SidebarProvider] viewType:', SidebarProvider.viewType);
    }
    resolveWebviewView(webviewView, _context, _token) {
        console.log('[Fedrag SidebarProvider] resolveWebviewView CALLED');
        console.log('[Fedrag SidebarProvider] webviewView.visible:', webviewView.visible);
        console.log('[Fedrag SidebarProvider] webviewView visible:', webviewView.visible);
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };
        webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);
        console.log('[Fedrag SidebarProvider] HTML set for webview');
        webviewView.webview.onDidReceiveMessage(async (data) => {
            console.log('[Fedrag SidebarProvider] Message received:', data?.type);
            if (data?.type !== 'sendMessage' || typeof data.text !== 'string') {
                return;
            }
            const userMessage = data.text.trim();
            if (!userMessage) {
                return;
            }
            const editor = vscode.window.activeTextEditor;
            const contextText = editor ? editor.document.getText() : '';
            try {
                const response = await axios_1.default.post('http://localhost:8000/recommend', {
                    message: userMessage,
                    context: contextText,
                });
                const markdown = response.data?.message ?? response.data ?? '';
                this._view?.webview.postMessage({ type: 'response', markdown });
            }
            catch (error) {
                const errorMessage = error?.response?.data?.error ??
                    error.message ??
                    'Request failed';
                this._view?.webview.postMessage({
                    type: 'error',
                    message: errorMessage,
                });
            }
        });
        console.log('[Fedrag SidebarProvider] resolveWebviewView COMPLETE');
    }
    getHtmlForWebview(webview) {
        const nonce = getNonce();
        const csp = [
            `default-src 'none'`,
            `style-src ${webview.cspSource} 'nonce-${nonce}'`,
            `script-src 'nonce-${nonce}' https://cdn.jsdelivr.net`,
            `img-src ${webview.cspSource} https: data:`,
            `font-src ${webview.cspSource}`,
            `connect-src https://cdn.jsdelivr.net`,
        ].join('; ');
        return /* html */ `
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta http-equiv="Content-Security-Policy" content="${csp}" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Fedrag Chat</title>
          <style nonce="${nonce}">
            :root {
              color-scheme: light dark;
            }
            body {
              padding: 0;
              margin: 0;
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              background: transparent;
              color: var(--vscode-editor-foreground);
            }
            .container {
              display: flex;
              flex-direction: column;
              height: 100vh;
            }
            #history {
              flex: 1;
              overflow-y: auto;
              padding: 12px;
              gap: 10px;
              display: flex;
              flex-direction: column;
            }
            .message {
              border-radius: 8px;
              padding: 10px 12px;
              border: 1px solid var(--vscode-input-border, #3c3c3c);
            }
            .message.user {
              background: var(--vscode-input-background);
            }
            .message.assistant {
              background: var(--vscode-editor-hoverHighlightBackground, #1e1e1e);
            }
            form {
              display: flex;
              gap: 8px;
              padding: 10px 12px;
              border-top: 1px solid var(--vscode-input-border, #3c3c3c);
            }
            input[type="text"] {
              flex: 1;
              padding: 8px;
              border-radius: 6px;
              border: 1px solid var(--vscode-input-border, #3c3c3c);
              background: var(--vscode-input-background);
              color: inherit;
            }
            button {
              padding: 8px 12px;
              border: none;
              border-radius: 6px;
              background: var(--vscode-button-background);
              color: var(--vscode-button-foreground);
              cursor: pointer;
            }
            button:hover {
              background: var(--vscode-button-hoverBackground);
            }
          </style>
        </head>
        <body>
          <div class="container">
            <div id="history" aria-live="polite"></div>
            <form id="chat-form">
              <input
                id="chat-input"
                type="text"
                placeholder="Ask Fedrag..."
                autocomplete="off"
              />
              <button type="submit">Send</button>
            </form>
          </div>
          <script nonce="${nonce}" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
          <script nonce="${nonce}">
            const vscode = acquireVsCodeApi();
            const history = document.getElementById('history');
            const form = document.getElementById('chat-form');
            const input = document.getElementById('chat-input');

            function appendMessage(role, content, isMarkdown = false) {
              const div = document.createElement('div');
              div.className = 'message ' + role;
              div.innerHTML = isMarkdown ? content : escapeHtml(content);
              history.appendChild(div);
              history.scrollTop = history.scrollHeight;
            }

            function escapeHtml(str) {
              return str.replace(/[&<>"']/g, (m) => {
                switch (m) {
                  case '&': return '&amp;';
                  case '<': return '&lt;';
                  case '>': return '&gt;';
                  case '"': return '&quot;';
                  case \"'\": return '&#39;';
                  default: return m;
                }
              });
            }

            form.addEventListener('submit', (event) => {
              event.preventDefault();
              const text = input.value.trim();
              if (!text) {
                return;
              }
              appendMessage('user', text);
              vscode.postMessage({ type: 'sendMessage', text });
              input.value = '';
              input.focus();
            });

            window.addEventListener('message', (event) => {
              const { type, markdown, message } = event.data;
              if (type === 'response') {
                const rendered = typeof marked !== 'undefined'
                  ? marked.parse(markdown || '')
                  : escapeHtml(markdown || '');
                appendMessage('assistant', rendered, true);
              } else if (type === 'error') {
                appendMessage('assistant', 'Error: ' + (message || 'Request failed'));
              }
            });
          </script>
        </body>
      </html>
    `;
    }
}
exports.SidebarProvider = SidebarProvider;
SidebarProvider.viewType = 'fedragSidebar';
function getNonce() {
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    return Array.from({ length: 32 }, () => possible.charAt(Math.floor(Math.random() * possible.length))).join('');
}
