import { Container } from '@theia/core/shared/inversify';
import { BackendApplication, BackendApplicationServer } from '@theia/core/lib/node/backend-application';
import { backendApplicationModule } from '@theia/core/lib/node/backend-application-module';
import { messagingBackendModule } from '@theia/core/lib/node/messaging/messaging-backend-module';
import { loggerBackendModule } from '@theia/core/lib/node/logger-backend-module';

import { fileSystemBackendModule } from '@theia/filesystem/lib/node/filesystem-backend-module';
import { workspaceBackendModule } from '@theia/workspace/lib/node/workspace-backend-module';
import { languagesBackendModule } from '@theia/languages/lib/node/languages-backend-module';
import { searchInWorkspaceBackendModule } from '@theia/search-in-workspace/lib/node/search-in-workspace-backend-module';
import { terminalBackendModule } from '@theia/terminal/lib/node/terminal-backend-module';
import { pluginBackendModule } from '@theia/plugin-ext/lib/plugin/node/plugin-backend-module';

export default new Container();

const container = new Container();
container.load(backendApplicationModule);
container.load(messagingBackendModule);
container.load(loggerBackendModule);

container.load(fileSystemBackendModule);
container.load(workspaceBackendModule);
container.load(languagesBackendModule);
container.load(searchInWorkspaceBackendModule);
container.load(terminalBackendModule);
container.load(pluginBackendModule);

function load(raw: Container): void {
    raw.load(backendApplicationModule);
}

export function start(port: number = 3000, host: string = 'localhost'): Promise<void> {
    const application = container.get<BackendApplication>(BackendApplication);
    const server = container.get<BackendApplicationServer>(BackendApplicationServer);
    return server.configure(port, host)
        .then(() => application.start())
        .then(() => {
            console.log(`Theia Backend running at http://${host}:${port}`);
        });
}

start();
