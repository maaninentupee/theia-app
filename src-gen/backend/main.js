// @ts-check
const { BackendApplicationConfigProvider } = require('@theia/core/lib/node/backend-application-config-provider');
const main = require('@theia/core/lib/node/main');

BackendApplicationConfigProvider.set({
    "singleInstance": true,
    "frontendConnectionTimeout": 0,
    "configurationFolder": ".theia"
});

globalThis.extensionInfo = [
    {
        "name": "@theia/core",
        "version": "1.57.1"
    },
    {
        "name": "@theia/variable-resolver",
        "version": "1.57.1"
    },
    {
        "name": "@theia/editor",
        "version": "1.57.1"
    },
    {
        "name": "@theia/callhierarchy",
        "version": "1.57.1"
    },
    {
        "name": "@theia/filesystem",
        "version": "1.57.1"
    },
    {
        "name": "@theia/process",
        "version": "1.57.1"
    },
    {
        "name": "@theia/workspace",
        "version": "1.57.1"
    },
    {
        "name": "@theia/file-search",
        "version": "1.57.1"
    },
    {
        "name": "@theia/navigator",
        "version": "1.57.1"
    },
    {
        "name": "@theia/markers",
        "version": "1.57.1"
    },
    {
        "name": "@theia/outline-view",
        "version": "1.57.1"
    },
    {
        "name": "@theia/monaco",
        "version": "1.57.1"
    },
    {
        "name": "@theia/scm",
        "version": "1.57.1"
    },
    {
        "name": "@theia/scm-extra",
        "version": "1.57.1"
    },
    {
        "name": "@theia/git",
        "version": "1.57.1"
    },
    {
        "name": "@theia/languages",
        "version": "1.4.0"
    },
    {
        "name": "@theia/json",
        "version": "1.3.0"
    },
    {
        "name": "@theia/userstorage",
        "version": "1.57.1"
    },
    {
        "name": "@theia/preferences",
        "version": "1.57.1"
    },
    {
        "name": "@theia/keymaps",
        "version": "1.57.1"
    },
    {
        "name": "@theia/messages",
        "version": "1.57.1"
    },
    {
        "name": "@theia/output",
        "version": "1.57.1"
    },
    {
        "name": "@theia/bulk-edit",
        "version": "1.57.1"
    },
    {
        "name": "@theia/console",
        "version": "1.57.1"
    },
    {
        "name": "@theia/terminal",
        "version": "1.57.1"
    },
    {
        "name": "@theia/task",
        "version": "1.57.1"
    },
    {
        "name": "@theia/test",
        "version": "1.57.1"
    },
    {
        "name": "@theia/debug",
        "version": "1.57.1"
    },
    {
        "name": "@theia/editor-preview",
        "version": "1.57.1"
    },
    {
        "name": "@theia/notebook",
        "version": "1.57.1"
    },
    {
        "name": "@theia/search-in-workspace",
        "version": "1.57.1"
    },
    {
        "name": "@theia/timeline",
        "version": "1.57.1"
    },
    {
        "name": "@theia/typehierarchy",
        "version": "1.57.1"
    },
    {
        "name": "@theia/plugin-ext",
        "version": "1.57.1"
    },
    {
        "name": "@theia/plugin-ext-vscode",
        "version": "1.57.1"
    },
    {
        "name": "@theia/mini-browser",
        "version": "1.57.1"
    },
    {
        "name": "@theia/preview",
        "version": "1.57.1"
    }
];

const serverModule = require('./server');
const serverAddress = main.start(serverModule());

serverAddress.then((addressInfo) => {
    if (process && process.send && addressInfo) {
        process.send(addressInfo);
    }
});

globalThis.serverAddress = serverAddress;
