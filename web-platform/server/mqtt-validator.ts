import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const ajv = new Ajv({ strict: false });
addFormats(ajv); // Add support for date-time, ipv4, uri, etc.
const schemas = new Map<string, any>();

/**
 * Topic-to-schema mapping
 * Maps MQTT topic patterns to their JSON schema files
 */
const TOPIC_SCHEMAS: Record<string, string> = {
  'devices/+/register': 'register-v1.0.0.json',
  'devices/+/heartbeat': 'heartbeat-v1.0.0.json',
  'devices/+/status': 'status-v1.0.0.json',
  'devices/+/snapshot': 'snapshot-v1.0.0.json',
  'devices/+/alerts': 'alerts-v1.0.0.json',
  'devices/+/webrtc/+': 'webrtc-signaling-v1.0.0.json',
  'devices/+/metrics': 'metrics-data-v1.0.0.json',
  'devices/+/commands/metrics': 'metrics-config-v1.0.0.json',
  'devices/+/commands/filter': 'filter-list-v1.0.0.json',
  'devices/+/commands/config': 'device-config-command-v1.0.0.json',
  'devices/+/logs': 'log-entry-v1.0.0.json',
  // Add more mappings as needed
};

/**
 * Load all JSON schemas on startup
 */
export async function initMQTTValidator(): Promise<void> {
  // __dirname is /app/server/ in Docker
  // Try Docker path first: /app/project-shared/schemas/
  // Fall back to local dev path: ../../shared/schemas/
  const dockerSchemasDir = join(__dirname, '../project-shared/schemas');
  const localSchemasDir = join(__dirname, '../../shared/schemas');
  const schemasDir = existsSync(dockerSchemasDir) ? dockerSchemasDir : localSchemasDir;
  
  // Load and register schemas that are referenced by $ref first
  // Detection schema is referenced by snapshot schema
  const referencedSchemas = ['detection-v1.0.0.json'];
  for (const schemaFile of referencedSchemas) {
    try {
      const fullPath = join(schemasDir, schemaFile);
      const schemaContent = await readFile(fullPath, 'utf-8');
      const schema = JSON.parse(schemaContent);
      ajv.addSchema(schema);
      console.log(`[MQTT Validator] ✓ Registered referenced schema: ${schemaFile}`);
    } catch (error) {
      console.warn(`[MQTT Validator] ⚠️  Failed to register ${schemaFile}:`, error);
    }
  }
  
  for (const [topicPattern, schemaPath] of Object.entries(TOPIC_SCHEMAS)) {
    try {
      const fullPath = join(schemasDir, schemaPath);
      const schemaContent = await readFile(fullPath, 'utf-8');
      const schema = JSON.parse(schemaContent);
      
      schemas.set(topicPattern, ajv.compile(schema));
      console.log(`[MQTT Validator] ✓ Loaded schema for ${topicPattern}`);
    } catch (error) {
      console.warn(`[MQTT Validator] ⚠️  Failed to load schema for ${topicPattern}:`, error);
    }
  }
  
  console.log(`[MQTT Validator] Initialized with ${schemas.size} schemas`);
}

/**
 * Validate MQTT message against its schema
 * @param topic - Full topic path (e.g., "devices/mac-dev-01/register")
 * @param payload - Parsed JSON payload
 * @returns true if valid
 * @throws Error if validation fails
 */
export function validateMQTTMessage(topic: string, payload: any): boolean {
  // ============================================================
  // DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
  // Validate all incoming MQTT messages before processing
  // Unknown topics and invalid payloads are rejected
  // ============================================================
  
  // Find matching schema by topic pattern
  const validator = findValidator(topic);
  
  if (!validator) {
    // No schema defined - HARD REJECTION for security
    const error = `No schema defined for topic: ${topic}`;
    console.error(`[MQTT Validator] 🚨 UNAUTHORIZED TOPIC: ${topic}`);
    throw new Error(error);
  }
  
  const valid = validator(payload);
  
  if (!valid) {
    const errors = validator.errors || [];
    console.error(`[MQTT Validator] 🚨 SCHEMA VALIDATION FAILED`);
    console.error(`[MQTT Validator] Topic: ${topic}`);
    console.error(`[MQTT Validator] Errors:`, JSON.stringify(errors, null, 2));
    throw new Error(`Schema validation failed: ${JSON.stringify(errors)}`);
  }
  
  // ============================================================
  // END SECURITY CRITICAL SECTION
  // ============================================================
  
  console.log(`[MQTT Validator] ✓ Valid message for ${topic}`);
  return true;
}

/**
 * Match topic to validator using wildcard patterns
 */
function findValidator(topic: string): any {
  for (const [pattern, validator] of schemas.entries()) {
    if (topicMatches(topic, pattern)) {
      return validator;
    }
  }
  return null;
}

/**
 * Check if topic matches pattern with wildcards
 * + matches single level
 */
function topicMatches(topic: string, pattern: string): boolean {
  const topicParts = topic.split('/');
  const patternParts = pattern.split('/');
  
  if (topicParts.length !== patternParts.length) {
    return false;
  }
  
  for (let i = 0; i < patternParts.length; i++) {
    const p = patternParts[i];
    const t = topicParts[i];
    
    if (p === '+') continue; // Wildcard matches anything
    if (p !== t) return false;
  }
  
  return true;
}
