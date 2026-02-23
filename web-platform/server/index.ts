import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import { setupMQTT } from "./mqtt";
import { setupRedis } from "./redis";
import * as OpenApiValidator from 'express-openapi-validator';
import $RefParser from '@apidevtools/json-schema-ref-parser';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// Parse JSON and URL-encoded bodies
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  // Initialize Redis
  log("🔴 Connecting to Redis...");
  await setupRedis();
  
  // Initialize MQTT
  log("📡 Connecting to MQTT broker...");
  const { initMQTTValidator } = await import('./mqtt-validator.js');
  await initMQTTValidator();
  setupMQTT();
  
  // Initialize Mediasoup (commented out for now - testing snapshot first)
  // log("🎥 Initializing Mediasoup media server...");
  // const { initMediasoup } = await import('./mediasoup-server.js');
  // await initMediasoup();
  
  // ============================================================
  // DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
  // OpenAPI validation middleware validates all REST API requests
  // Must be placed BEFORE routes to intercept all requests
  // ============================================================
  // Pre-load and dereference OpenAPI spec once on startup
  // This inlines all $ref schemas to avoid duplicate $id registration in AJV
  const dockerSpecPath = join(__dirname, '../project-shared/specs/openapi.yaml');
  const localSpecPath = join(__dirname, '../../shared/specs/openapi.yaml');
  const apiSpecPath = existsSync(dockerSpecPath) ? dockerSpecPath : localSpecPath;
  
  log("📋 Loading and dereferencing OpenAPI spec from: " + apiSpecPath);
  const dereferencedSpec = await $RefParser.dereference(apiSpecPath) as any;
  
  // Strip $id fields from dereferenced schemas to prevent AJV duplicate registration
  // When a schema is referenced multiple times (e.g., register-v1.0.0.json in both
  // Device and DeviceCreate), the inlined copies still have the same $id, causing
  // "already exists" errors. Removing $id makes them anonymous inline schemas.
  function stripSchemaIds(obj: any): void {
    if (typeof obj !== 'object' || obj === null) return;
    if (obj.$id) delete obj.$id;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        stripSchemaIds(obj[key]);
      }
    }
  }
  stripSchemaIds(dereferencedSpec);
  
  log("✅ OpenAPI spec dereferenced and $id fields stripped");
  
  app.use(
    OpenApiValidator.middleware({
      apiSpec: dereferencedSpec, // Pass dereferenced object instead of file path
      validateRequests: {
        allowUnknownQueryParameters: false,
        coerceTypes: true, // Convert string "123" to number 123
        removeAdditional: false, // Don't silently remove unknown properties
      },
      validateResponses: process.env.NODE_ENV === 'development', // Validate in dev only
      validateSecurity: false, // We use custom bearer token auth middleware
      ignorePaths: /^(?!\/api)/, // Only validate /api/* paths
    })
  );
  log("✅ OpenAPI validation middleware enabled");
  // ============================================================
  // END SECURITY CRITICAL SECTION
  // ============================================================
  
  const server = await registerRoutes(app);

  // Global error handler - MUST be after routes
  app.use((err: any, req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";
    
    // OpenAPI route not found in spec - FAIL FAST
    if (err.status === 404 && err.errors) {
      console.error('\n' + '='.repeat(80));
      console.error('❌ FATAL: OpenAPI Validation Failure');
      console.error('='.repeat(80));
      console.error(`Route not documented in OpenAPI spec: ${req.method} ${req.path}`);
      console.error('');
      console.error('This endpoint exists in code but is missing from:');
      console.error('  shared/specs/openapi.yaml');
      console.error('');
      console.error('All API routes MUST be documented in the OpenAPI specification.');
      console.error('Add this route to openapi.yaml before continuing.');
      console.error('');
      console.error('Error details:');
      console.error(`  Status: ${err.status}`);
      console.error(`  Message: ${err.message}`);
      if (err.errors) {
        console.error(`  Validation errors: ${JSON.stringify(err.errors, null, 2)}`);
      }
      console.error('='.repeat(80) + '\n');
      
      // Send error response before crashing
      res.status(404).json({
        error: 'Route not in OpenAPI specification',
        route: `${req.method} ${req.path}`,
        message: 'This endpoint is not documented. Server will terminate.'
      });
      
      // Crash the server to enforce OpenAPI compliance
      setTimeout(() => {
        process.exit(1);
      }, 100);
      return;
    }
    
    // OpenAPI validation errors (400)
    if (err.status === 400 && err.errors) {
      log(`❌ Validation error on ${req.method} ${req.path}: ${JSON.stringify(err.errors)}`);
      return res.status(400).json({ 
        error: 'Validation failed', 
        details: err.errors.map((e: any) => ({
          path: e.path,
          message: e.message,
          errorCode: e.errorCode
        }))
      });
    }

    res.status(status).json({ message });
    if (status >= 500) {
      console.error(err);
    }
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || '5000', 10);
  server.listen(port, "0.0.0.0", () => {
    log(`serving on port ${port}`);
  });
})();
