/**
 * Don't touch this file. It will be regenerated by theia build.
 * To customize webpack configuration change C:\Users\Tony Keltakangas\CascadeProjects\theia-app\webpack.config.js
 */
// @ts-check
const path = require('path');
const webpack = require('webpack');
const yargs = require('yargs');
const resolvePackagePath = require('resolve-package-path');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin')
const MiniCssExtractPlugin = require('mini-css-extract-plugin')

const outputPath = path.resolve(__dirname, 'lib', 'frontend');
const { mode, staticCompression }  = yargs.option('mode', {
    description: "Mode to use",
    choices: ["development", "production"],
    default: "production"
}).option('static-compression', {
    description: 'Controls whether to enable compression of static artifacts.',
    type: 'boolean',
    default: true
}).argv;
const development = mode === 'development';

const plugins = [
    new CopyWebpackPlugin({
        patterns: [
            {
                // copy secondary window html file to lib folder
                from: path.resolve(__dirname, 'src-gen/frontend/secondary-window.html')
            },
            {
                // copy webview files to lib folder
                from: path.join(resolvePackagePath('@theia/plugin-ext', __dirname), '..', 'src', 'main', 'browser', 'webview', 'pre'),
                to: path.resolve(__dirname, 'lib', 'webview', 'pre')
            }
            ,
            {
                // copy frontend plugin host files
                from: path.join(resolvePackagePath('@theia/plugin-ext-vscode', __dirname), '..', 'lib', 'node', 'context', 'plugin-vscode-init-fe.js'),
                to: path.resolve(__dirname, 'lib', 'frontend', 'context', 'plugin-vscode-init-fe.js')
            }
        ]
    }),
    new webpack.ProvidePlugin({
        // the Buffer class doesn't exist in the browser but some dependencies rely on it
        Buffer: ['buffer', 'Buffer']
    })
];
// it should go after copy-plugin in order to compress monaco as well
if (staticCompression) {
    plugins.push(new CompressionPlugin({}));
}

module.exports = [{
    mode,
    plugins,
    devtool: 'source-map',
    entry: {
        bundle: path.resolve(__dirname, 'src-gen/frontend/index.js'),
        'editor.worker': '@theia/monaco-editor-core/esm/vs/editor/editor.worker.js'
    },
    output: {
        filename: '[name].js',
        path: outputPath,
        devtoolModuleFilenameTemplate: 'webpack:///[resource-path]?[loaders]',
        globalObject: 'self'
    },
    target: 'web',
    cache: staticCompression,
    module: {
        rules: [
            {
                test: /\.css$/,
                exclude: /materialcolors\.css$|\.useable\.css$/,
                use: ['style-loader', 'css-loader']
            },
            {
                test: /materialcolors\.css$|\.useable\.css$/,
                use: [
                    {
                        loader: 'style-loader',
                        options: {
                            esModule: false,
                            injectType: 'lazySingletonStyleTag',
                            attributes: {
                                id: 'theia-theme'
                            }
                        }
                    },
                    'css-loader'
                ]
            },
            {
                test: /\.(ttf|eot|svg)(\?v=\d+\.\d+\.\d+)?$/,
                type: 'asset',
                parser: {
                    dataUrlCondition: {
                        maxSize: 10000,
                    }
                },
                generator: {
                    dataUrl: {
                        mimetype: 'image/svg+xml'
                    }
                }
            },
            {
                test: /\.(jpg|png|gif)$/,
                type: 'asset/resource',
                generator: {
                    filename: '[hash].[ext]'
                }
            },
            {
                // see https://github.com/eclipse-theia/theia/issues/556
                test: /source-map-support/,
                loader: 'ignore-loader'
            },
            {
                test: /\.js$/,
                enforce: 'pre',
                loader: 'source-map-loader',
                exclude: /jsonc-parser|fast-plist|onigasm/
            },
            {
                test: /\.woff(2)?(\?v=[0-9]\.[0-9]\.[0-9])?$/,
                type: 'asset',
                parser: {
                    dataUrlCondition: {
                        maxSize: 10000,
                    }
                },
                generator: {
                    dataUrl: {
                        mimetype: 'image/svg+xml'
                    }
                }
            },
            {
                test: /node_modules[\\|/](vscode-languageserver-types|vscode-uri|jsonc-parser|vscode-languageserver-protocol)/,
                loader: 'umd-compat-loader'
            },
            {
                test: /\.wasm$/,
                type: 'asset/resource'
            },
            {
                test: /\.plist$/,
                type: 'asset/resource'
            }
        ]
    },
    resolve: {
        fallback: {
            'child_process': false,
            'crypto': false,
            'net': false,
            'path': require.resolve('path-browserify'),
            'process': false,
            'os': false,
            'timers': false
        },
        extensions: ['.js']
    },
    stats: {
        warnings: true,
        children: true
    },
    ignoreWarnings: [
        // Some packages do not have source maps, that's ok
        /Failed to parse source map/,
        {
            // Monaco uses 'require' in a non-standard way
            module: /@theia\/monaco-editor-core/,
            message: /require function is used in a way in which dependencies cannot be statically extracted/
        }
    ]
},
{
    mode,
    plugins: [
        new MiniCssExtractPlugin({
            // Options similar to the same options in webpackOptions.output
            // both options are optional
            filename: "[name].css",
            chunkFilename: "[id].css",
        }),
    ],
    devtool: 'source-map',
    entry: {
        "secondary-window": path.resolve(__dirname, 'src-gen/frontend/secondary-index.js'),
    },
    output: {
        filename: '[name].js',
        path: outputPath,
        devtoolModuleFilenameTemplate: 'webpack:///[resource-path]?[loaders]',
        globalObject: 'self'
    },
    target: 'web',
    cache: staticCompression,
    module: {
        rules: [
            {
                test: /.css$/i,
                use: [MiniCssExtractPlugin.loader, "css-loader"]
            },
            {
                test: /.wasm$/,
                type: 'asset/resource'
            }
        ]
    },
    resolve: {
        fallback: {
            'child_process': false,
            'crypto': false,
            'net': false,
            'path': require.resolve('path-browserify'),
            'process': false,
            'os': false,
            'timers': false
        },
        extensions: ['.js']
    },
    stats: {
        warnings: true,
        children: true
    },
    ignoreWarnings: [
        {
            // Monaco uses 'require' in a non-standard way
            module: /@theia\/monaco-editor-core/,
            message: /require function is used in a way in which dependencies cannot be statically extracted/
        }
    ]
}];