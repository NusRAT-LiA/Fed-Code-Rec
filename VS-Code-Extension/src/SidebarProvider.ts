// import * as vscode from 'vscode';
// import axios from 'axios';

// export class SidebarProvider implements vscode.WebviewViewProvider {
//   public static readonly viewType = 'fedragSidebar';

//   private _view?: vscode.WebviewView;

//   constructor(private readonly _extensionUri: vscode.Uri) {
//     console.log('[Fedrag SidebarProvider] Constructor called');
//     console.log('[Fedrag SidebarProvider] viewType:', SidebarProvider.viewType);
//   }

//   resolveWebviewView(
//     webviewView: vscode.WebviewView,
//     _context: vscode.WebviewViewResolveContext,
//     _token: vscode.CancellationToken
//   ): void {
//     console.log('[Fedrag SidebarProvider] resolveWebviewView CALLED');
//     console.log('[Fedrag SidebarProvider] webviewView.visible:', webviewView.visible);
//     console.log('[Fedrag SidebarProvider] webviewView visible:', webviewView.visible);
    
//     this._view = webviewView;

//     webviewView.webview.options = {
//       enableScripts: true,
//       localResourceRoots: [this._extensionUri],
//     };

//     webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);
//     console.log('[Fedrag SidebarProvider] HTML set for webview');

//     webviewView.webview.onDidReceiveMessage(async (data) => {
//       console.log('[Fedrag SidebarProvider] Message received:', data?.type);
//       if (data?.type !== 'sendMessage' || typeof data.text !== 'string') {
//         return;
//       }

//       const userMessage = data.text.trim();
//       if (!userMessage) {
//         return;
//       }

//       const editor = vscode.window.activeTextEditor;
//       const contextText = editor ? editor.document.getText() : '';

//       try {
//         const response = await axios.post('http://localhost:8000/recommend', {
//           message: userMessage,
//           context: contextText,
//         });

//         const markdown = response.data?.message ?? response.data ?? '';
//         this._view?.webview.postMessage({ type: 'response', markdown });
//       } catch (error: unknown) {
//         const errorMessage =
//           (error as any)?.response?.data?.error ??
//           (error as Error).message ??
//           'Request failed';

//         this._view?.webview.postMessage({
//           type: 'error',
//           message: errorMessage,
//         });
//       }
//     });

//     console.log('[Fedrag SidebarProvider] resolveWebviewView COMPLETE');
//   }

//   private getHtmlForWebview(webview: vscode.Webview): string {
//     const nonce = getNonce();
//     const logoUri = webview.asWebviewUri(
//       vscode.Uri.joinPath(this._extensionUri, 'media', 'icon.svg')
//     );

//     const csp = [
//       `default-src 'none'`,
//       `style-src ${webview.cspSource} 'nonce-${nonce}'`,
//       `script-src 'nonce-${nonce}' https://cdn.jsdelivr.net`,
//       `img-src ${webview.cspSource} https: data:`,
//       `font-src ${webview.cspSource}`,
//       `connect-src https://cdn.jsdelivr.net`,
//     ].join('; ');

//     return /* html */ `
//       <!DOCTYPE html>
//       <html lang="en">
//         <head>
//           <meta charset="UTF-8" />
//           <meta http-equiv="Content-Security-Policy" content="${csp}" />
//           <meta name="viewport" content="width=device-width, initial-scale=1.0" />
//           <title>Fedrag Chat</title>
//           <style nonce="${nonce}">
//             :root {
//                 color-scheme: light dark;

//                 /* Pastel green palette */
//                 --green-bg-soft: #e9f7ef;
//                 --green-bg-user: #d1f2e0;
//                 --green-bg-assistant: #f3fbf7;
//                 --green-border: #a8e0c2;
//                 --green-accent: #5fbf90;
//                 --green-accent-hover: #4ea87d;
//               }

//             body {
//               padding: 0;
//               margin: 0;
//               font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
//               background: transparent;
//               color: var(--vscode-editor-foreground);
//             }
//             .container {
//               display: flex;
//               flex-direction: column;
//               height: 100vh;
//             }
//             .header {
//               display: flex;
//               align-items: center;
//               gap: 10px;
//               padding: 10px 12px 4px 12px;
//               border-bottom: 1px solid var(--vscode-input-border, #3c3c3c);
//             }
//             .header-logo {
//               width: 28px;
//               height: 28px;
//               flex-shrink: 0;
//             }
//             .header-logo img {
//               width: 100%;
//               height: 100%;
//               object-fit: contain;
//             }
//             .header-text {
//               display: flex;
//               flex-direction: column;
//               gap: 2px;
//             }
//             .header-title {
//               font-size: 13px;
//               font-weight: 600;
//             }
//             .header-subtitle {
//               font-size: 11px;
//               opacity: 0.8;
//             }
//             .welcome {
//               padding: 6px 12px 8px 12px;
//               font-size: 11px;
//               border-bottom: 1px solid var(--vscode-input-border, #3c3c3c);
//             }
//             .welcome-steps {
//               margin: 4px 0 0 0;
//               padding-left: 16px;
//             }
//             .welcome-steps li {
//               margin-bottom: 2px;
//             }
//             #history {
//               flex: 1;
//               overflow-y: auto;
//               padding: 10px 12px 12px 12px;
//               gap: 10px;
//               display: flex;
//               flex-direction: column;
//             }
//             .message {
//               border-radius: 8px;
//               padding: 10px 12px;
//               border: 1px solid var(--vscode-input-border, #3c3c3c);
//             }
//             .message.user {
//               background: var(--vscode-input-background);
//             }
//             .message.assistant {
//               background: var(--vscode-editor-hoverHighlightBackground, #1e1e1e);
//             }
//             form {
//               display: flex;
//               gap: 8px;
//               padding: 10px 12px;
//               border-top: 1px solid var(--vscode-input-border, #3c3c3c);
//             }
//             input[type="text"] {
//               flex: 1;
//               padding: 8px;
//               border-radius: 6px;
//               border: 1px solid var(--vscode-input-border, #3c3c3c);
//               background: var(--vscode-input-background);
//               color: inherit;
//             }
//             button {
//               padding: 8px 12px;
//               border: none;
//               border-radius: 6px;
//               background: var(--vscode-button-background);
//               color: var(--vscode-button-foreground);
//               cursor: pointer;
//             }
//             button:hover {
//               background: var(--vscode-button-hoverBackground);
//             }
//           </style>
//         </head>
//         <body>
//           <div class="container">
//             <div class="header">
//               <div class="header-logo" aria-hidden="true">
//                 <img src="${logoUri}" alt="Fedrag logo" />
//               </div>
//               <div class="header-text">
//                 <div class="header-title">Fedrag Assistant</div>
//                 <div class="header-subtitle">Chat with context from your current file</div>
//               </div>
//             </div>
//             <section class="welcome" aria-label="How to use Fedrag">
//               <div><strong>Welcome!</strong> Here’s how to use Fedrag effectively:</div>
//               <ol class="welcome-steps">
//                 <li>Open the file or selection you want help with.</li>
//                 <li>Type a question about that code, error, or refactor idea.</li>
//                 <li>Review the response and copy edits back into your editor.</li>
//               </ol>
//             </section>
//             <div id="history" aria-live="polite"></div>
//             <form id="chat-form">
//               <input
//                 id="chat-input"
//                 type="text"
//                 placeholder="Ask Fedrag..."
//                 autocomplete="off"
//               />
//               <button type="submit">Send</button>
//             </form>
//           </div>
//           <script nonce="${nonce}" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
//           <script nonce="${nonce}">
//             const vscode = acquireVsCodeApi();
//             const history = document.getElementById('history');
//             const form = document.getElementById('chat-form');
//             const input = document.getElementById('chat-input');

//             function appendMessage(role, content, isMarkdown = false) {
//               const div = document.createElement('div');
//               div.className = 'message ' + role;
//               div.innerHTML = isMarkdown ? content : escapeHtml(content);
//               history.appendChild(div);
//               history.scrollTop = history.scrollHeight;
//             }

//             function escapeHtml(str) {
//               return str.replace(/[&<>"']/g, (m) => {
//                 switch (m) {
//                   case '&': return '&amp;';
//                   case '<': return '&lt;';
//                   case '>': return '&gt;';
//                   case '"': return '&quot;';
//                   case \"'\": return '&#39;';
//                   default: return m;
//                 }
//               });
//             }

//             form.addEventListener('submit', (event) => {
//               event.preventDefault();
//               const text = input.value.trim();
//               if (!text) {
//                 return;
//               }
//               appendMessage('user', text);
//               vscode.postMessage({ type: 'sendMessage', text });
//               input.value = '';
//               input.focus();
//             });

//             window.addEventListener('message', (event) => {
//               const { type, markdown, message } = event.data;
//               if (type === 'response') {
//                 const rendered = typeof marked !== 'undefined'
//                   ? marked.parse(markdown || '')
//                   : escapeHtml(markdown || '');
//                 appendMessage('assistant', rendered, true);
//               } else if (type === 'error') {
//                 appendMessage('assistant', 'Error: ' + (message || 'Request failed'));
//               }
//             });
//           </script>
//         </body>
//       </html>
//     `;
//   }
// }

// function getNonce(): string {
//   const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
//   return Array.from({ length: 32 }, () => possible.charAt(Math.floor(Math.random() * possible.length))).join('');
// }

import * as vscode from 'vscode';
import axios from 'axios';

export class SidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'fedragSidebar';

  private _view?: vscode.WebviewView;

  constructor(private readonly _extensionUri: vscode.Uri) {
    // Constructor logic
  }

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (data) => {
      if (data?.type !== 'sendMessage' || typeof data.text !== 'string') {
        return;
      }

      const userMessage = data.text.trim();
      if (!userMessage) return;

      const editor = vscode.window.activeTextEditor;
      const contextText = editor ? editor.document.getText() : '';

      // Send a temporary "thinking" status if you wanted to add that UI state later
      // this._view?.webview.postMessage({ type: 'thinking' });

      try {
        const response = await axios.post('http://localhost:8000/recommend', {
          message: userMessage,
          context: contextText,
        });

        const markdown = response.data?.message ?? response.data ?? '';
        this._view?.webview.postMessage({ type: 'response', markdown });
      } catch (error: unknown) {
        const errorMessage =
          (error as any)?.response?.data?.error ??
          (error as Error).message ??
          'Request failed';

        this._view?.webview.postMessage({
          type: 'error',
          message: errorMessage,
        });
      }
    });
  }

  private getHtmlForWebview(webview: vscode.Webview): string {
    const nonce = getNonce();
    const logoUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'icon.svg')
    );

    // Content Security Policy
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
            /* --- RESET & VARIABLES --- */
            :root {
              --bg-color: #e8f5e9;           /* Pastel Green Background */
              --chat-bg-user: #4caf50;       /* Darker Green for User */
              --chat-text-user: #ffffff;     /* White text for User */
              --chat-bg-ai: #ffffff;         /* White card for AI */
              --chat-text-ai: #2c2c2c;       /* Dark Grey for AI */
              --input-bg: #ffffff;
              --border-color: #c8e6c9;
              --shadow-sm: 0 1px 2px rgba(0,0,0,0.1);
              --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
            }

            * { box-sizing: border-box; }

            body {
              padding: 0;
              margin: 0;
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
              background-color: var(--bg-color);
              color: #333;
              height: 100vh;
              overflow: hidden; /* Prevent body scroll, handle in containers */
            }

            /* --- LAYOUT --- */
            .container {
              display: flex;
              flex-direction: column;
              height: 100%;
            }

            /* --- HEADER --- */
            .header {
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 16px 20px;
              background: rgba(255,255,255,0.6);
              backdrop-filter: blur(5px);
              border-bottom: 1px solid var(--border-color);
              flex-shrink: 0;
            }
            .header-logo {
              width: 32px;
              height: 32px;
              background: white;
              border-radius: 8px;
              padding: 4px;
              box-shadow: var(--shadow-sm);
            }
            .header-logo img { width: 100%; height: 100%; object-fit: contain; }
            .header-text { display: flex; flex-direction: column; }
            .header-title { font-size: 14px; font-weight: 700; color: #1b5e20; }
            .header-subtitle { font-size: 11px; color: #66bb6a; font-weight: 500; }

            /* --- CHAT HISTORY --- */
            #history {
              flex: 1;
              overflow-y: auto;
              padding: 20px;
              display: flex;
              flex-direction: column;
              gap: 16px;
              scroll-behavior: smooth;
            }

            /* --- MESSAGES --- */
            .message-wrapper {
              display: flex;
              width: 100%;
              opacity: 0; /* For animation */
              animation: fadeIn 0.3s forwards;
            }
            
            @keyframes fadeIn { to { opacity: 1; transform: translateY(0); } }

            .message-wrapper.user { justify-content: flex-end; }
            .message-wrapper.assistant { justify-content: flex-start; }

            .message {
              max-width: 85%;
              padding: 12px 16px;
              border-radius: 12px;
              font-size: 13px;
              line-height: 1.5;
              box-shadow: var(--shadow-sm);
              word-wrap: break-word;
            }

            /* User Styles */
            .message.user {
              background: var(--chat-bg-user);
              color: var(--chat-text-user);
              border-bottom-right-radius: 2px; /* Chat bubble effect */
            }

            /* Assistant Styles */
            .message.assistant {
              background: var(--chat-bg-ai);
              color: var(--chat-text-ai);
              border-bottom-left-radius: 2px; /* Chat bubble effect */
              border: 1px solid #e0f2f1;
            }

            .message.error {
              background: #ffebee;
              color: #c62828;
              border: 1px solid #ef9a9a;
            }

            /* --- CODE BLOCKS (Markdown Styling) --- */
            /* Important: Style code within AI messages to look like an editor */
            .message pre {
              background: #2d2d2d; /* Dark editor bg */
              color: #f8f8f2;
              padding: 12px;
              border-radius: 6px;
              overflow-x: auto;
              margin: 8px 0;
              font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
              font-size: 12px;
            }
            .message code {
              font-family: 'Consolas', monospace;
              background: rgba(0,0,0,0.06);
              padding: 2px 4px;
              border-radius: 4px;
            }
            .message.user code {
              background: rgba(255,255,255,0.2); /* Lighter bg for user code */
            }
            .message pre code {
              background: transparent;
              padding: 0;
              color: inherit;
            }

            /* --- INPUT AREA --- */
            .input-container {
              padding: 16px;
              background: transparent; /* Let body bg show through */
            }
            
            #chat-form {
              display: flex;
              gap: 8px;
              background: white;
              padding: 6px;
              border-radius: 24px; /* Capsule shape */
              box-shadow: var(--shadow-md);
              border: 1px solid var(--border-color);
              transition: border 0.2s, box-shadow 0.2s;
            }

            #chat-form:focus-within {
              border-color: #4caf50;
              box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
            }

            input[type="text"] {
              flex: 1;
              border: none;
              outline: none;
              padding: 8px 12px;
              font-size: 13px;
              background: transparent;
              color: #333;
            }

            button {
              background: #4caf50;
              color: white;
              border: none;
              border-radius: 20px;
              padding: 8px 16px;
              font-weight: 600;
              font-size: 12px;
              cursor: pointer;
              transition: background 0.2s;
            }
            
            button:hover {
              background: #388e3c;
            }

            /* Welcome Screen Styling */
            .welcome {
              text-align: center;
              padding: 40px 20px;
              color: #558b2f;
              opacity: 0.8;
            }
            .welcome-steps {
              text-align: left;
              background: white;
              padding: 15px 15px 15px 35px;
              border-radius: 8px;
              margin-top: 15px;
              box-shadow: var(--shadow-sm);
              color: #333;
            }
          </style>
        </head>
        <body>
          <div class="container">
            
            <div class="header">
              <div class="header-logo">
                <img src="${logoUri}" alt="Fedrag" />
              </div>
              <div class="header-text">
                <div class="header-title">Fedrag Assistant</div>
                <div class="header-subtitle">Your AI Code Companion</div>
              </div>
            </div>

            <div id="history">
              <div class="welcome">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 8px;">Hello!</div>
                <div>I can help you refactor, explain, or fix your code.</div>
                <ol class="welcome-steps">
                  <li>Highlight some code or open a file.</li>
                  <li>Ask me a question below.</li>
                  <li>I'll analyze it using the backend.</li>
                </ol>
              </div>
            </div>

            <div class="input-container">
              <form id="chat-form">
                <input
                  id="chat-input"
                  type="text"
                  placeholder="Ask something..."
                  autocomplete="off"
                />
                <button type="submit">Send</button>
              </form>
            </div>

          </div>

          <script nonce="${nonce}" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
          <script nonce="${nonce}">
            const vscode = acquireVsCodeApi();
            const history = document.getElementById('history');
            const form = document.getElementById('chat-form');
            const input = document.getElementById('chat-input');
            const welcomeMsg = document.querySelector('.welcome');

            // Handle scrolling
            function scrollToBottom() {
              history.scrollTop = history.scrollHeight;
            }

            function appendMessage(role, content, isMarkdown = false) {
              // Hide welcome message on first chat
              if (welcomeMsg) welcomeMsg.style.display = 'none';

              const wrapper = document.createElement('div');
              wrapper.className = 'message-wrapper ' + role;

              const bubble = document.createElement('div');
              bubble.className = 'message ' + role;
              
              if (isMarkdown && typeof marked !== 'undefined') {
                bubble.innerHTML = marked.parse(content);
              } else {
                bubble.textContent = content; // safer than innerHTML for raw text
              }

              wrapper.appendChild(bubble);
              history.appendChild(wrapper);
              scrollToBottom();
            }

            form.addEventListener('submit', (event) => {
              event.preventDefault();
              const text = input.value.trim();
              if (!text) return;

              appendMessage('user', text);
              vscode.postMessage({ type: 'sendMessage', text });
              
              input.value = '';
              input.focus();
            });

            window.addEventListener('message', (event) => {
              const { type, markdown, message } = event.data;
              if (type === 'response') {
                appendMessage('assistant', markdown || '', true);
              } else if (type === 'error') {
                appendMessage('error', message || 'Unknown error');
              }
            });
          </script>
        </body>
      </html>
    `;
  }
}

function getNonce(): string {
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  return Array.from({ length: 32 }, () => possible.charAt(Math.floor(Math.random() * possible.length))).join('');
}