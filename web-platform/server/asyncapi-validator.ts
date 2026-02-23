import { Parser } from '@asyncapi/parser';
import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import Ajv from 'ajv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let parsedSpec: any = null;
const ajv = new Ajv({ allErrors: true, strict: false });

/**
 * Initialize AsyncAPI validator
 * Call this once on server startup
 */
export async function initAsyncApiValidator(): Promise<void> {
  const specPath = join(__dirname, '../project-shared/specs/asyncapi.yaml');
  console.log('[AsyncAPI] Loading spec from:', specPath);
  
  const specContent = await readFile(specPath, 'utf-8');
  const parser = new Parser();
  
  const { document, diagnostics } = await parser.parse(specContent);
  
  if (diagnostics.length > 0) {
    console.warn('[AsyncAPI] Spec has diagnostics (non-fatal):', diagnostics.map(d => d.message));
    // Don't throw - we can still validate messages without full $ref resolution
  }
  
  parsedSpec = document;
  console.log('[AsyncAPI] Specification loaded successfully');
}

/**
 * Validate MQTT message against AsyncAPI spec
 * @param topic MQTT topic (e.g. "devices/mac-dev-01/register")
 * @param payload Message payload (already parsed JSON)
 * @returns true if valid, throws error if invalid
 */
export function validateMessage(topic: string, payload: any): boolean {
  if (!parsedSpec) {
    throw new Error('AsyncAPI validator not initialized. Call initAsyncApiValidator() first.');
  }
  
  // Find matching channel by pattern
  const channel = findMatchingChannel(topic);
  if (!channel) {
    console.warn(`[AsyncAPI] No channel definition found for topic: ${topic}`);
    return true; // Don't fail on unknown topics - log warning only
  }
  
  // Get message schema
  const messageSchema = getMessageSchema(channel, payload);
  if (!messageSchema) {
    console.warn(`[AsyncAPI] No message schema found for topic: ${topic}`);
    return true;
  }
  
  // Validate payload against schema
  const validate = ajv.compile(messageSchema);
  const valid = validate(payload);
  
  if (!valid) {
    const errors = validate.errors || [];
    console.error(`[AsyncAPI] Validation failed for topic ${topic}:`, errors);
    throw new Error(`Invalid message for topic ${topic}: ${JSON.stringify(errors)}`);
  }
  
  console.log(`[AsyncAPI] ✓ Valid message for topic: ${topic}`);
  return true;
}

/**
 * Find channel that matches the given topic
 * Handles wildcard patterns like devices/{deviceId}/register
 */
function findMatchingChannel(topic: string): any {
  const channels = parsedSpec.json().channels || {};
  
  for (const [channelName, channel] of Object.entries(channels)) {
    const address = (channel as any).address;
    if (!address) continue;
    
    // Convert AsyncAPI pattern to regex
    // devices/{deviceId}/register -> devices/[^/]+/register
    const pattern = address.replace(/{[^}]+}/g, '[^/]+');
    const regex = new RegExp(`^${pattern}$`);
    
    if (regex.test(topic)) {
      return channel;
    }
  }
  
  return null;
}

/**
 * Extract message schema from channel definition
 */
function getMessageSchema(channel: any, payload: any): any {
  const messages = channel.messages || {};
  
  // Get first message schema (most channels have one message type)
  const messageKey = Object.keys(messages)[0];
  if (!messageKey) return null;
  
  const message = messages[messageKey];
  const payloadDef = message.payload;
  
  if (!payloadDef) return null;
  
  // If it's a $ref, resolve it
  if (payloadDef.$ref) {
    return resolveRef(payloadDef.$ref);
  }
  
  return payloadDef;
}

/**
 * Resolve $ref to actual schema
 */
function resolveRef(ref: string): any {
  // $ref format: #/components/schemas/DeviceRegister
  const parts = ref.split('/').filter(p => p !== '#');
  
  let current = parsedSpec.json();
  for (const part of parts) {
    current = current[part];
    if (!current) {
      console.warn(`[AsyncAPI] Could not resolve $ref: ${ref}`);
      return null;
    }
  }
  
  return current;
}
