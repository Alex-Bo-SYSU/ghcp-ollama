import express from 'express';
import bodyParser from 'body-parser';
import { CopilotLSPClient } from './lsp-client.js';
import { CopilotAuth } from './auth.js';
import { CopilotModels } from './models.js';
import { CopilotChat } from './chat.js';

// Create Express app
const app = express();
app.use(bodyParser.json());

// Global variables
let lspClient = null;
let auth = null;
let models = null;
let chat = null;
let authRefreshInterval = null;
const PORT = process.env.PORT || 11434; // Same port as Ollama

// Initialize the Copilot client
async function initializeClient() {
  if (lspClient) {
    return { success: true };
  }

  try {
    console.log('Initializing GitHub Copilot LSP client...');
    lspClient = new CopilotLSPClient({
      onNotification: (method, params) => {
        if (method === 'statusNotification' && params.status === 'Error') {
          console.error(`Copilot status error: ${params.message}`);
        }
      }
    });

    await lspClient.start();

    auth = new CopilotAuth(lspClient);
    models = new CopilotModels(lspClient);
    chat = new CopilotChat(lspClient);

    // Check authentication status
    const status = await auth.checkStatus();

    if (!status.user) {
      console.log('Not authenticated. Please run the auth command first or make a request to /api/auth/signin');
      return {
        success: false,
        message: 'Authentication required',
        needsAuth: true
      };
    }

    // Set up auto token refresh - check every hour if token is still valid
    startAuthRefresh();

    return { success: true, user: status.user };
  } catch (error) {
    console.error('Failed to initialize Copilot client:', error);
    return {
      success: false,
      message: error.message
    };
  }
}

// Auto-refresh authentication token
function startAuthRefresh() {
  if (authRefreshInterval) {
    clearInterval(authRefreshInterval);
  }

  // Check every hour
  authRefreshInterval = setInterval(async () => {
    try {
      const status = await auth.checkStatus();
      if (!status.user) {
        console.log('Authentication token expired. Attempting to re-authenticate...');
        // If there's a stored token that can be used for re-auth, implement that here
        // For now, we'll just log that re-authentication is needed
        console.log('Re-authentication required. Please use /api/auth/signin endpoint.');
      } else {
        console.log(`Authentication still valid for user: ${status.user}`);
      }
    } catch (error) {
      console.error('Error checking authentication status:', error);
    }
  }, 60 * 60 * 1000); // 1 hour
}

// Middleware to ensure the client is initialized
async function ensureClientInitialized(req, res, next) {
  if (!lspClient) {
    const initResult = await initializeClient();

    if (!initResult.success) {
      // If it needs authentication specifically, return 401
      if (initResult.needsAuth) {
        return res.status(401).json({
          error: 'Authentication required',
          message: 'Please authenticate using the /api/auth/signin endpoint'
        });
      }

      // Otherwise return a 500 error
      return res.status(500).json({
        error: 'Failed to initialize GitHub Copilot client',
        message: initResult.message
      });
    }
  }
  next();
}

// Ollama API endpoints
app.get('/api/tags', ensureClientInitialized, async (req, res) => {
  try {
    const modelsResult = await models.getAvailableModels();

    if (modelsResult.success) {
      // Format the response to match Ollama's format
      const ollamaResponse = {
        models: modelsResult.availableModels.map(model => ({
          name: model.id,
          modified_at: new Date().toISOString(),
          size: 0, // Size is not applicable for Copilot models
          digest: `copilot-${model.id}`,
          details: {
            parameter_size: "unknown",
            family: "GitHub Copilot",
            families: ["GitHub Copilot"],
            format: "Copilot API",
            description: model.description
          }
        }))
      };

      return res.json(ollamaResponse);
    } else {
      return res.status(500).json({
        error: 'Failed to get models',
        message: modelsResult.error
      });
    }
  } catch (error) {
    console.error('Error getting models:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Handle chat requests, supports streaming
async function handleChatRequest(req, res) {
  const { model, messages, stream, options } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    console.error('Invalid request: Missing or empty messages array');
    return res.status(400).json({
      error: 'Bad request',
      message: 'Messages array is required and must not be empty'
    });
  }

  try {
    if (stream) {
      console.log('Using streaming mode for response');
      // Set headers for SSE
      // res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      console.log('Set SSE headers');

      // Send SSE headers
      res.write('\n');

      let isDone = false;

      console.log('Sending message to chat service...');
      const chatResult = await chat.sendMessage(messages, {
        model: model,
        temperature: options?.temperature || 0.5,
        maxResults: 1
      }, (response) => {
        let createdAt = new Date();
        let respModel = "";
        if (response.done) {
          console.log('Received done signal, sending final SSE message');
          // Send final message
          const sseData = {
            model: respModel,
            created_at: createdAt.toISOString(),
            message: {
              role: 'assistant',
              content: ''
            },
            done: true,
            total_duration: 4883583458,
            load_duration: 1334875,
            prompt_eval_count: 26,
            prompt_eval_duration: 342546000,
            eval_count: 282,
            eval_duration: 4535599000
          };

          res.write(`${JSON.stringify(sseData)}\n\n`);
          isDone = true;
          console.log('Ending response stream');
          res.end();
        } else {
          createdAt = new Date(response.created * 1000);
          respModel = response.model;
          // Create SSE message in Ollama format
          const sseData = {
            model: response.model,
            created_at: createdAt.toISOString(),
            message: response.message,
            done: false,
          };

          res.write(`${JSON.stringify(sseData)}\n\n`);
          res.flush && res.flush(); // Force flush if available
        }
      });

      console.log('Chat sendMessage result:', JSON.stringify(chatResult));

      if (!chatResult.success) {
        console.error(`Chat request failed: ${chatResult.error}`);
        if (!isDone) {
          // If we haven't sent a done message yet, send an error
          const sseData = {
            error: 'Failed to generate text',
            message: chatResult.error
          };
          res.write(`data: ${JSON.stringify(sseData)}\n\n`);
          res.end();
        }
      }

      // Handle client disconnect
      req.on('close', () => {
        console.log('Client disconnected, cleaning up resources');
        // Clean up any resources if needed
      });
    } else {
      console.log('Using non-streaming mode for response');
      // Non-streaming mode - collect complete response
      let fullResponse = '';

      console.log('Sending message to chat service...');
      const chatResult = await chat.sendMessage(prompt, {
        model: model,
        temperature: options?.temperature || 0.5,
        maxResults: 1
      }, (response) => {
        if (response.type === 'solution') {
          // Keep track of the full response
          fullResponse = response.content;
          console.log(`Received solution, current length: ${fullResponse.length} chars`);
        } else if (response.type === 'done') {
          console.log('Received done signal for non-streaming response');
        }
      });

      console.log('Chat sendMessage result:', JSON.stringify(chatResult));

      if (!chatResult.success) {
        console.error(`Chat request failed: ${chatResult.error}`);
        return res.status(500).json({
          error: 'Failed to generate text',
          message: chatResult.error
        });
      }

      console.log('Waiting for completion...');
      // Wait for completion
      await new Promise((resolve) => {
        const checkInterval = setInterval(() => {
          if (fullResponse) {
            console.log(`Response complete, final length: ${fullResponse.length} chars`);
            clearInterval(checkInterval);
            resolve();
          } else {
            console.log('Still waiting for response...');
          }
        }, 100);

        // Timeout after 30 seconds
        setTimeout(() => {
          console.log('Response timeout after 30 seconds');
          clearInterval(checkInterval);
          resolve();
        }, 30000);
      });

      // Format response like Ollama
      console.log('Sending final JSON response');
      return res.json({
        model,
        created_at: new Date().toISOString(),
        message: {
          role: 'assistant',
          content: fullResponse
        },
        done: true,
        total_duration: 0,
        load_duration: 0,
        prompt_eval_duration: 0,
        eval_count: 0,
        eval_duration: 0
      });
    }
  } catch (error) {
    console.error('Error in chat request:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
}

// Also add logging to the chat class in chat.js
app.post('/api/chat', ensureClientInitialized, (req, res) => {
  console.log('API call to /api/chat received');
  return handleChatRequest(req, res);
});

app.get('/api/token', ensureClientInitialized, async (req, res) => {
  try {
    const tempChat = new CopilotChat(lspClient);
    const credentials = await tempChat._getCredentials();
    
    if (!credentials || !credentials.token) {
      return res.status(401).json({
        success: false,
        error: 'No valid GitHub Copilot token found'
      });
    }

    return res.json({
      success: true,
      token: credentials.token
    });
    
  } catch (error) {
    console.error('Error getting token:', error);
    return res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Add enhanced error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled server error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message,
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`GitHub Copilot Ollama server running on port ${PORT}`);

  // Initialize client on startup
  initializeClient().then(result => {
    if (result.success) {
      console.log('GitHub Copilot client initialized successfully');
    } else {
      console.log('GitHub Copilot client initialization failed:', result.message);
      if (result.needsAuth) {
        console.log('Please authenticate using the /api/auth/signin endpoint');
      }
    }
  }).catch(error => {
    console.error('Error during initialization:', error);
  });
});

// Handle shutdown gracefully
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

function shutdown() {
  console.log('Shutting down server...');

  if (authRefreshInterval) {
    clearInterval(authRefreshInterval);
  }

  if (lspClient) {
    lspClient.stop();
  }

  process.exit(0);
}
