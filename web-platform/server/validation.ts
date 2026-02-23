/**
 * Message validation using JSON Schema.
 * Validates messages against schemas in /shared/schemas/
 */
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Initialize AJV with strict mode for security
const ajv = new Ajv({
  strict: true,
  allErrors: true,
  verbose: true,
});
addFormats(ajv);

// Load schema from shared/schemas directory
function loadSchema(schemaPath: string) {
  // __dirname is /app/server/ in Docker
  // In Docker: /app/project-shared/schemas/
  // In local dev: ../../shared/schemas/ (from web-platform/server/)
  const projectSharedPath = join(__dirname, '../project-shared/schemas', schemaPath);
  const localSharedPath = join(__dirname, '../../shared/schemas', schemaPath);
  
  // Try Docker path first
  if (existsSync(projectSharedPath)) {
    const schema = JSON.parse(readFileSync(projectSharedPath, 'utf-8'));
    return schema;
  }
  
  // Fall back to local dev path
  if (existsSync(localSharedPath)) {
    const schema = JSON.parse(readFileSync(localSharedPath, 'utf-8'));
    return schema;
  }
  
  // Neither path exists - provide helpful error
  throw new Error(
    `Schema not found: ${schemaPath}\n` +
    `Tried:\n` +
    `  - ${projectSharedPath}\n` +
    `  - ${localSharedPath}`
  );
}

// Load and register schemas that are referenced by $ref in other schemas
// This must be done before compiling schemas that reference them
const connectionAckSchema = loadSchema('connection-ack-v1.0.0.json');
const detectionSchema = loadSchema('detection-v1.0.0.json');
const snapshotSchema = loadSchema('snapshot-v1.0.0.json');
const metricsDataSchema = loadSchema('metrics-data-v1.0.0.json');
const logCommandSchema = loadSchema('log-command-v1.0.0.json');
const logEntrySchema = loadSchema('log-entry-v1.0.0.json');
const websocketLogStartSchema = loadSchema('websocket-log-start-v1.0.0.json');
const alertsSchema = loadSchema('alerts-v1.0.0.json');

// Add schemas to AJV's schema store using their $id URIs (compile will auto-register them)
ajv.addSchema(connectionAckSchema);
ajv.addSchema(detectionSchema);
ajv.addSchema(snapshotSchema);
ajv.addSchema(metricsDataSchema);
ajv.addSchema(logCommandSchema);
ajv.addSchema(logEntrySchema);
ajv.addSchema(websocketLogStartSchema);
ajv.addSchema(alertsSchema);

// Compile and cache validators
const validators = {
  // WebSocket messages use envelope schema (references the schemas added above)
  websocketEnvelope: ajv.compile(loadSchema('websocket-envelope-v1.0.0.json')),
  
  // Individual payload validators (get from AJV cache by $id instead of re-compiling)
  connected: ajv.getSchema(connectionAckSchema.$id!)!,
  snapshot: ajv.getSchema(snapshotSchema.$id!)!,
  metricsData: ajv.getSchema(metricsDataSchema.$id!)!,
  logCommand: ajv.getSchema(logCommandSchema.$id!)!,
  logEntry: ajv.getSchema(logEntrySchema.$id!)!,
  alert: ajv.getSchema(alertsSchema.$id!)!,
  
  // MQTT (for server-side validation when bridging)
  register: ajv.compile(loadSchema('register-v1.0.0.json')),
  heartbeat: ajv.compile(loadSchema('heartbeat-v1.0.0.json')),
  status: ajv.compile(loadSchema('status-v1.0.0.json')),
  alerts: ajv.getSchema(alertsSchema.$id!)!,
  metricsCommand: ajv.compile(loadSchema('metrics-config-v1.0.0.json')),
};

export interface ValidationResult {
  valid: boolean;
  errors?: string[];
}

/**
 * Validate WebSocket message from client
 */
export function validateClientMessage(message: any): ValidationResult {
  if (!message || typeof message !== 'object') {
    return { valid: false, errors: ['Message must be an object'] };
  }

  // Validate against envelope schema
  const valid = validators.websocketEnvelope(message);
  if (!valid) {
    return {
      valid: false,
      errors: validators.websocketEnvelope.errors?.map(err => `${err.instancePath} ${err.message}`) || ['Validation failed'],
    };
  }

  return { valid: true };
}

/**
 * Validate WebSocket message to client (server → browser)
 */
export function validateServerMessage(message: any): ValidationResult {
  if (!message || typeof message !== 'object') {
    return { valid: false, errors: ['Message must be an object'] };
  }

  // Validate against envelope schema
  const valid = validators.websocketEnvelope(message);
  if (!valid) {
    return {
      valid: false,
      errors: validators.websocketEnvelope.errors?.map(err => `${err.instancePath} ${err.message}`) || ['Validation failed'],
    };
  }

  return { valid: true };
}

/**
 * Validate MQTT message (for server-side validation)
 */
export function validateMQTTMessage(topic: string, payload: any): ValidationResult {
  if (!payload || typeof payload !== 'object') {
    return { valid: false, errors: ['Payload must be an object'] };
  }

  // Extract message type from topic
  const topicParts = topic.split('/');
  const messageType = topicParts[topicParts.length - 1];
  
  let validator;
  switch (messageType) {
    case 'register':
      validator = validators.register;
      break;
    case 'heartbeat':
      validator = validators.heartbeat;
      break;
    case 'status':
      validator = validators.status;
      break;
    case 'alerts':
      validator = validators.alerts;
      break;
    case 'snapshot':
      validator = validators.snapshot;
      break;
    case 'metrics':
      // For metrics topic, could be either config or data
      // Try data first as it's more common
      validator = validators.metricsData;
      if (!validator(payload)) {
        // Try config schema
        validator = validators.metricsCommand;
      }
      break;
    case 'logs':
      validator = validators.logEntry;
      break;
    default:
      // Unknown topic, skip validation
      return { valid: true };
  }

  const valid = validator(payload);
  if (!valid) {
    return {
      valid: false,
      errors: validator.errors?.map(err => `${err.instancePath} ${err.message}`) || ['Validation failed'],
    };
  }

  return { valid: true };
}

// Groq response validator (cached to avoid duplicate schema registration)
let groqResponseValidator: ReturnType<typeof ajv.compile> | null = null;

/**
 * Validate Groq CV response against schema
 */
export function validateGroqResponse(response: unknown): { valid: boolean; errors?: string[] } {
  // Lazy initialization - compile validator only once
  if (!groqResponseValidator) {
    const schemaPath = 'groq-cv-response-v1.0.0.json';
    const schema = loadSchema(schemaPath);
    groqResponseValidator = ajv.compile(schema);
  }
  
  const valid = groqResponseValidator(response);
  
  if (!valid) {
    return {
      valid: false,
      errors: groqResponseValidator.errors?.map(err => `${err.instancePath} ${err.message}`) || ['Validation failed'],
    };
  }
  
  return { valid: true };
}
