export class CopilotChat {
  constructor(lspClient) {
    this.lspClient = lspClient;
    this.callbacks = {};
    this.documents = {};
  }

  /**
   * Send a chat message to Copilot and stream the response
   * @param {Object} messages - The message to send, in format of chat history
   * @param {Object} options - Chat options
   * @param {function} onResponse - Callback for streaming responses
   */
  async sendMessage(messages, options = {}, onResponse) {
    try {
      // Step 1: Check authentication status and refresh token if needed
      // Use the auth module to check status and handle token refresh
      const { CopilotAuth } = await import('./auth.js');
      const auth = new CopilotAuth(this.lspClient);

      const status = await auth.checkStatus();

      if (!status.authenticated) {
        // If token is expired but we can refresh it
        if (status.lspAuthenticated && status.tokenExpired) {
          console.log('GitHub token expired, attempting to refresh...');
          const refreshed = await auth.fetchAndStoreGitHubToken();

          if (!refreshed) {
            return {
              success: false,
              error: 'GitHub token expired and refresh failed. Please sign in again.'
            };
          }

          console.log('GitHub token refreshed successfully');
        } else if (!status.lspAuthenticated) {
          return {
            success: false,
            error: 'Not authenticated with GitHub Copilot. Please sign in first.'
          };
        } else {
          return {
            success: false,
            error: status.error || 'Authentication issue: ' + JSON.stringify(status)
          };
        }
      }

      // Step 2: Send HTTP request to the Copilot API endpoint
      const result = await this._sendChatRequest(messages, options, onResponse);

      return result;
    } catch (error) {
      console.error('Error sending chat messages:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Send a direct HTTP request to the Copilot chat API
   * @param {Object} messages - The message to send, in format of chat history
   * @param {Object} options - Chat options
   * @param {function} onResponse - Callback for streaming responses
   * @private
   */
  async _sendChatRequest(messages, options = {}, onResponse) {
    // Get credentials (similar to models.js)
    const credentials = await this._getCredentials();
    if (!credentials || !credentials.token) {
      return {
        success: false,
        error: 'No GitHub Copilot credentials found. Please sign in.'
      };
    }

    // Get the API endpoint - for chat it would be similar to the models endpoint
    const apiEndpoint = await this._getApiEndpoint();
    if (!apiEndpoint) {
      return {
        success: false,
        error: 'Could not determine API endpoint'
      };
    }

    return new Promise(async (resolve, reject) => {
      try {
        // Get the default model from model-config.json
        const defaultModel = await this._getDefaultModel();

        // Parse the API endpoint URL
        const url = new URL(`${apiEndpoint}/chat/completions`);

        // Prepare the request payload
        const payload = {
          model: options.model || defaultModel,
          messages: messages,
          stream: true,
          temperature: options.temperature || 0.5,
          max_tokens: options.maxTokens || 8192
        };
        
        console.log("url:", url);

        const requestOptions = {
          hostname: url.hostname,
          path: url.pathname,
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${credentials.token}`,
            'Content-Type': 'application/json',
            'Copilot-Integration-Id': 'vscode-chat',
            'Editor-Version': 'Neovim/0.10.3'
          }
        };

        console.log("requestOptions:", requestOptions);

        // Use https or http depending on the protocol
        const requestModule = url.protocol === 'https:' ?
          await import('https') : await import('http');

        const req = requestModule.default.request(requestOptions, (res) => {
          if (res.statusCode !== 200) {
            let errorData = '';
            res.on('data', (chunk) => {
              errorData += chunk;
            });

            res.on('end', () => {
              console.error(`API returned status code ${res.statusCode}: ${errorData}`);
              reject(new Error(`API returned status code ${res.statusCode}: ${errorData}`));
            });

            return;
          }

          // Handle SSE (Server-Sent Events) response
          let buffer = '';

          res.on('data', (chunk) => {
            buffer += chunk.toString();
            // Process complete SSE messages
            const resp_msgs = buffer.split('\n\n');
            buffer = resp_msgs.pop();

            for (const resp_msg of resp_msgs) {
              if (!resp_msg || resp_msg.trim() === '') continue;

              // Parse the SSE data line
              const lines = resp_msg.split('\n');
              let data = '';

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  data = line.slice(6); // Remove 'data: ' prefix
                  if (data === '[DONE]') {
                    // console.log("receive [DONE] in on data");
                    // if (onResponse && typeof onResponse === 'function') {
                    //   onResponse({
                    //     done: true,
                    //     message: {
                    //       role: 'assistant',
                    //       content: '',
                    //     },
                    //   });
                    // }
                    break;
                  }
                  try {
                    const parsed = JSON.parse(data);

                    if (parsed.choices && parsed.choices[0]) {
                      const choice = parsed.choices[0];
                      // console.log("Receive choice:", choice);

                      if (choice.delta) {
                        // Send the accumulated text so far
                        if (onResponse && typeof onResponse === 'function') {
                          onResponse({
                            done: false,
                            message: {
                              role: 'assistant',
                              content: choice.delta.content ?? ""
                            },
                            model: parsed.model,
                            created: parsed.created
                          });
                        }
                      }
                    }
                  } catch (error) {
                    console.error('Error parsing SSE data:', error);
                    console.error('SSE data:', data);
                  }
                }
              }
            }
          });

          res.on('end', () => {
            console.log("Receive end");
            if (onResponse && typeof onResponse === 'function') {
              onResponse({
                done: true,
                message: {
                  role: 'assistant',
                  content: '',
                },
                model: undefined,
                created: undefined,
              });
            }

            resolve({ success: true });
          });
        });

        req.on('error', (error) => {
          console.error('Error during API request:', error);
          reject(error);
        });

        // Write the JSON payload to the request
        req.write(JSON.stringify(payload));
        req.end();

      } catch (error) {
        console.error('Error in _sendChatRequest:', error);
        reject(error);
      }
    });
  }

  /**
   * Get the default model from model-config.json
   * @private
   */
  async _getDefaultModel() {
    try {
      const fs = await import('fs');
      const path = await import('path');
      const os = await import('os');

      // Define the config path based on XDG spec, fallback to default locations
      const configPath = process.env.XDG_CONFIG_HOME ||
        (os.platform() === 'win32'
          ? path.join(os.homedir(), 'AppData', 'Local', 'github-copilot')
          : path.join(os.homedir(), '.config', 'github-copilot'));

      const modelConfigPath = path.join(configPath, 'model-config.json');

      // Check if the model config file exists
      if (fs.existsSync(modelConfigPath)) {
        try {
          const configContent = fs.readFileSync(modelConfigPath, 'utf8');
          const configData = JSON.parse(configContent);

          if (configData && configData.id) {
            console.log(`Using default model from model-config.json: ${configData.id}`);
            return configData.id;
          }
        } catch (error) {
          console.error('Error reading model-config.json:', error);
        }
      }

      // Default to gpt-4o if no model config found
      console.log('No model config found, using default model: gpt-4o');
      return 'gpt-4o';
    } catch (error) {
      console.error('Error getting default model:', error);
      return 'gpt-4o';
    }
  }

  /**
   * Get the API endpoint for Copilot
   * @private
   */
  async _getApiEndpoint() {
    try {
      const fs = await import('fs');
      const path = await import('path');
      const os = await import('os');

      // Define the config path based on XDG spec, fallback to default locations
      const configPath = process.env.XDG_CONFIG_HOME ||
        (os.platform() === 'win32'
          ? path.join(os.homedir(), 'AppData', 'Local', 'github-copilot')
          : path.join(os.homedir(), '.config', 'github-copilot'));

      const githubTokenPath = path.join(configPath, 'github-token.json');

      // Try to read the GitHub token file to get the API endpoint
      if (fs.existsSync(githubTokenPath)) {
        try {
          const tokenContent = fs.readFileSync(githubTokenPath, 'utf8');
          const tokenData = JSON.parse(tokenContent);

          if (tokenData.endpoints && tokenData.endpoints.api) {
            console.log(`Using API endpoint from github-token.json: ${tokenData.endpoints.api}`);
            return tokenData.endpoints.api;
          }
        } catch (error) {
          console.error('Error reading github-token.json for API endpoint:', error);
        }
      }

      // Fallback to the default endpoint
      console.log('Using default API endpoint: https://api.githubcopilot.com');
      return 'https://api.githubcopilot.com';
    } catch (error) {
      console.error('Error getting API endpoint:', error);
      // Fallback to the default endpoint
      return 'https://api.githubcopilot.com';
    }
  }

  /**
   * Get the GitHub Copilot credentials - updated to use github-token.json
   * @private
   */
  async _getCredentials() {
    try {
      const fs = await import('fs');
      const path = await import('path');
      const os = await import('os');

      // Define the config path based on XDG spec, fallback to default locations
      const configPath = process.env.XDG_CONFIG_HOME ||
        (os.platform() === 'win32'
          ? path.join(os.homedir(), 'AppData', 'Local', 'github-copilot')
          : path.join(os.homedir(), '.config', 'github-copilot'));

      const githubTokenPath = path.join(configPath, 'github-token.json'); // Changed to use dash instead of underscore

      // First try to get the token from github-token.json
      if (fs.existsSync(githubTokenPath)) {
        try {
          const tokenContent = fs.readFileSync(githubTokenPath, 'utf8');
          const tokenData = JSON.parse(tokenContent);

          // The token is directly in the token field of the JSON
          if (tokenData.token) {
            console.log('Using GitHub token from github-token.json');
            return {
              token: tokenData.token,
              username: tokenData.user
            };
          }
        } catch (error) {
          console.error('Error reading github-token.json:', error);
        }
      }
      return null;
    } catch (error) {
      console.error('Error getting credentials:', error);
      return null;
    }
  }
}

export default CopilotChat;
