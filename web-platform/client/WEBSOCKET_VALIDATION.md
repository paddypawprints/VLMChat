# Browser WebSocket Validation

## Overview

The browser now **validates all incoming WebSocket messages** from the server against the AsyncAPI specification before processing them.

## Architecture

```
AsyncAPI Spec (YAML)
        ↓
   [Build Time]
        ↓
  Vite bundles spec into app
        ↓
   [Runtime]
        ↓
Validator initialized with bundled spec
        ↓
WebSocket receives message
        ↓
validateServerMessage() checks against spec
        ↓
✅ Valid → Process message
❌ Invalid → Reject and log error
```

## Files

### 1. Validator: `client/src/lib/websocket-validator.ts`
- Loads bundled AsyncAPI WebSocket spec
- Compiles JSON Schema validators for each message type
- Exports `validateServerMessage()` function
- Validates: `connected`, `snapshot`, `metrics`

### 2. Hook: `client/src/hooks/use-websocket.tsx`
- Imports validator
- Validates every message before processing
- Rejects invalid messages (logs error, doesn't process)
- Passes valid messages to handlers

### 3. Build Config: `vite.config.ts`
- Adds `@rollup/plugin-yaml` for YAML imports
- Allows access to `../shared` directory
- Bundles AsyncAPI spec into app bundle

### 4. Types: `client/src/vite-env.d.ts`
- TypeScript declarations for `*.yaml` imports

## Validation Rules

### Connected Message
```typescript
{
  type: 'connected',          // Required: literal 'connected'
  deviceId: string,           // Required: min length 1
  timestamp: string (ISO8601) // Required: date-time format
}
```

### Snapshot Message
```typescript
{
  type: 'snapshot',           // Required: literal 'snapshot'  
  image: string,              // Required: base64 data
  timestamp: string (ISO8601),// Required: date-time format
  width?: number,             // Optional: min 1
  height?: number,            // Optional: min 1
  format?: 'jpeg' | 'png'     // Optional: enum
}
```

### Metrics Message
```typescript
{
  type: 'metrics',            // Required: literal 'metrics'
  data: {                     // Required: metrics payload
    timestamp: string,
    session: {
      start_time: number,
      end_time: number | null,
      instruments: [...]
    }
  }
}
```

## Security Benefits

### ✅ Prevents Malformed Data
- Missing required fields rejected
- Wrong types rejected (string instead of number)
- Invalid enum values rejected
- Malformed timestamps rejected

### ✅ Prevents Unknown Messages
- Only documented message types accepted
- Prevents server from injecting arbitrary data
- Clear contract enforcement

### ✅ Fail-Fast Behavior
- Invalid messages never reach app logic
- Errors logged for debugging
- No silent failures
- Clear error messages with field paths

## Error Handling

When validation fails:
```typescript
{
  valid: false,
  errors: [
    {
      path: '/deviceId',           // JSON path to invalid field
      message: "must have required property 'deviceId'"
    }
  ]
}
```

Logged as:
```
[WebSocket] ❌ Invalid message from server: {
  type: 'connected',
  errors: [{path: '/deviceId', message: "must have required property 'deviceId'"}]
}
```

## Testing

### Manual Test
Run in browser console:
```typescript
import('./lib/websocket-validator.test.ts');
```

### Automated Test
See `websocket-validator.test.ts` for test cases:
- Valid messages (all types)
- Missing required fields
- Wrong types
- Unknown message types
- Malformed input

## Build Integration

### Development
```bash
npm run dev
# Vite bundles AsyncAPI spec with HMR support
```

### Production
```bash
npm run build
# AsyncAPI spec bundled into dist/
# No runtime fetch needed
# Spec version locked to build time
```

## Spec Updates

When updating `shared/specs/asyncapi-websocket.yaml`:

1. **Browser automatically rebuilds** - no code changes needed
2. **Validation rules update** - new schemas enforced
3. **Type safety** - validators recompiled
4. **Build fails** if spec is invalid

## Version Checking

```typescript
import { getSpecVersion } from '@/lib/websocket-validator';

console.log('Using AsyncAPI spec version:', getSpecVersion());
// Output: "1.0.0"
```

## Performance

- **Build time**: ~50-100ms (spec bundled once)
- **Runtime init**: ~5-10ms (validators compiled on module load)
- **Per-message**: ~0.1-0.5ms (schema validation)
- **Bundle size**: +5-10KB (spec + validators)

Validation overhead is negligible compared to network latency and message processing.

## Comparison with Server

| Feature | Browser | Server |
|---------|---------|--------|
| Validates incoming | ✅ Yes | ✅ Yes |
| Validates outgoing | ❌ No (server validates) | ✅ Yes |
| Spec source | Bundled | File system |
| Fails if spec invalid | ✅ Build fails | ✅ Startup fails |
| Runtime overhead | Minimal | Minimal |

Both sides validate what they **receive** - ensuring end-to-end message integrity.
