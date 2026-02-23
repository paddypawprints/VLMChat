/**
 * WebSocket message validation for browser client.
 * Validates all incoming messages from server against websocket-envelope schema.
 */
import Ajv from 'ajv';
import addFormats from 'ajv-formats';

// Import JSON schemas using @shared alias (configured in vite.config.ts)
import websocketEnvelopeSchema from '@shared/schemas/websocket-envelope-v1.0.0.json';
import connectionAckSchema from '@shared/schemas/connection-ack-v1.0.0.json';
import detectionSchema from '@shared/schemas/detection-v1.0.0.json';
import snapshotSchema from '@shared/schemas/snapshot-v1.0.0.json';
import metricsDataSchema from '@shared/schemas/metrics-data-v1.0.0.json';
import logEntrySchema from '@shared/schemas/log-entry-v1.0.0.json';
import websocketLogStartSchema from '@shared/schemas/websocket-log-start-v1.0.0.json';
import alertsSchema from '@shared/schemas/alerts-v1.0.0.json';

const ajv = new Ajv({
  strict: true,
  allErrors: true,
  verbose: true,
});
addFormats(ajv);

// Add referenced schemas to AJV's schema store (detection must be added before snapshot)
ajv.addSchema(connectionAckSchema);
ajv.addSchema(detectionSchema);
ajv.addSchema(snapshotSchema);
ajv.addSchema(metricsDataSchema);
ajv.addSchema(logEntrySchema);
ajv.addSchema(alertsSchema);
ajv.addSchema(websocketLogStartSchema);

// Compile validators
const validators: Record<string, any> = {
  // Envelope validator for all WebSocket messages
  envelope: ajv.compile(websocketEnvelopeSchema),
  
  // Individual payload validators (for additional validation if needed)
  connected: ajv.compile(connectionAckSchema),
  snapshot: ajv.compile(snapshotSchema),
  metrics: ajv.compile(metricsDataSchema),
  logs: ajv.compile(logEntrySchema),
};

console.log('[WebSocket Validator] Initialized with schemas:', Object.keys(validators));

export interface ValidationResult {
  valid: boolean;
  errors?: Array<{
    path: string;
    message: string;
  }>;
}

/**
 * Validate incoming WebSocket message from server
 */
export function validateServerMessage(message: any): ValidationResult {
  if (!message || typeof message !== 'object') {
    return {
      valid: false,
      errors: [{ path: '', message: 'Message must be an object' }]
    };
  }

  // Validate against envelope schema
  const valid = validators.envelope(message);

  if (!valid) {
    return {
      valid: false,
      errors: (validators.envelope.errors || []).map((err: any) => ({
        path: err.instancePath || err.dataPath,
        message: err.message || 'Validation failed'
      }))
    };
  }

  return { valid: true };
}

/**
 * Get schema version for debugging
 */
export function getSchemaVersion(): string {
  return websocketEnvelopeSchema.$id || 'unknown';
}
