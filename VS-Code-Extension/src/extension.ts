import * as vscode from 'vscode';
import { SidebarProvider } from './SidebarProvider';

export function activate(context: vscode.ExtensionContext): void {
  console.log('[Fedrag Extension] ========== ACTIVATE CALLED ==========');
  console.log('[Fedrag Extension] ViewType:', SidebarProvider.viewType);
  console.log('[Fedrag Extension] Expected view ID from package.json: fedragSidebar');
  console.log('[Fedrag Extension] Extension URI:', context.extensionUri.toString());

  const provider = new SidebarProvider(context.extensionUri);
  console.log('[Fedrag Extension] SidebarProvider created');

  // Register provider synchronously and immediately - MUST happen before VS Code tries to resolve
  const viewTypeToRegister = SidebarProvider.viewType;
  console.log('[Fedrag Extension] Registering provider for viewType:', viewTypeToRegister);
  console.log('[Fedrag Extension] ViewType matches expected ID?', viewTypeToRegister === 'fedragSidebar');
  
  try {
    const registration = vscode.window.registerWebviewViewProvider(
      viewTypeToRegister,
      provider,
      {
        webviewOptions: {
          retainContextWhenHidden: true
        }
      }
    );
    console.log('[Fedrag Extension] ✅ registerWebviewViewProvider SUCCESS');
    console.log('[Fedrag Extension] Registered viewType:', viewTypeToRegister);
    console.log('[Fedrag Extension] Registration Disposable:', registration);
    context.subscriptions.push(registration);
    
    // Verify registration by trying to get the view
    setTimeout(() => {
      console.log('[Fedrag Extension] Checking if view can be resolved...');
      vscode.commands.executeCommand('workbench.view.extension.fedrag').then(
        () => {
          console.log('[Fedrag Extension] View command executed successfully');
        },
        (err: unknown) => {
          console.error('[Fedrag Extension] Error executing view command:', err);
        }
      );
    }, 100);
  } catch (error) {
    console.error('[Fedrag Extension] ERROR registering provider:', error);
    console.error('[Fedrag Extension] Error stack:', (error as Error).stack);
  }

  context.subscriptions.push(
    vscode.commands.registerCommand('fedrag.start', async () => {
      console.log('[Fedrag Extension] Command fedrag.start executed');
      // Reveal the Fedrag view container so users can see the chat.
      await vscode.commands.executeCommand('workbench.view.extension.fedrag');
    })
  );

  console.log('[Fedrag Extension] Activation complete');
}

export function deactivate(): void {
  // Nothing to clean up explicitly.
}

