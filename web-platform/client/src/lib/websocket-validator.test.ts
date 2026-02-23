/**
 * Test file to verify WebSocket validator is working
 * Run this in browser console after app loads
 */

import { validateServerMessage, getSpecVersion } from './websocket-validator';

console.log('WebSocket Validator Test Suite');
console.log('AsyncAPI Spec Version:', getSpecVersion());

// Test 1: Valid connected message
const validConnected = {
  type: 'connected',
  deviceId: 'test-device',
  timestamp: new Date().toISOString()
};
console.log('Test 1 - Valid connected:', validateServerMessage(validConnected));

// Test 2: Invalid connected message (missing deviceId)
const invalidConnected = {
  type: 'connected',
  timestamp: new Date().toISOString()
};
console.log('Test 2 - Invalid connected (missing deviceId):', validateServerMessage(invalidConnected));

// Test 3: Valid snapshot message
const validSnapshot = {
  type: 'snapshot',
  image: 'base64data',
  timestamp: new Date().toISOString(),
  width: 1920,
  height: 1080,
  format: 'jpeg'
};
console.log('Test 3 - Valid snapshot:', validateServerMessage(validSnapshot));

// Test 4: Invalid snapshot (missing required field)
const invalidSnapshot = {
  type: 'snapshot',
  timestamp: new Date().toISOString()
};
console.log('Test 4 - Invalid snapshot (missing image):', validateServerMessage(invalidSnapshot));

// Test 5: Unknown message type
const unknownType = {
  type: 'unknown_message_type',
  data: 'test'
};
console.log('Test 5 - Unknown type:', validateServerMessage(unknownType));

// Test 6: Not an object
console.log('Test 6 - Not an object:', validateServerMessage('invalid'));

export {};
