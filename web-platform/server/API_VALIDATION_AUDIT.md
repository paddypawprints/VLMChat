# REST API Validation Audit

## Summary

**Status**: ❌ **CRITICAL GAPS FOUND**

The browser makes numerous REST API calls but **most endpoints lack runtime validation** against the OpenAPI spec.

## Findings

### ✅ Endpoints Documented in OpenAPI

| Method | Endpoint | OpenAPI | Server Code | Validation |
|--------|----------|---------|-------------|------------|
| POST | `/api/auth/login` | ✅ | ✅ | ❌ No |
| GET | `/api/auth/me` | ✅ | ✅ | ❌ No |
| POST | `/api/auth/logout` | ✅ | ✅ | ❌ No |
| GET | `/api/devices` | ✅ | ✅ | ❌ No |
| POST | `/api/devices` | ✅ | ✅ | ❌ No |
| GET | `/api/devices/{deviceId}` | ✅ | ✅ | ❌ No |
| DELETE | `/api/devices/{deviceId}` | ✅ | ✅ | ❌ No |
| GET | `/api/devices/{deviceId}/messages` | ✅ | ✅ | ❌ No |
| POST | `/api/devices/{deviceId}/snapshot` | ✅ | ✅ | ❌ No |
| GET | `/api/devices/{deviceId}/snapshot` | ✅ | ✅ | ❌ No |
| GET | `/api/devices/{deviceId}/metrics` | ✅ | ✅ | ❌ No |
| GET | `/api/devices/{deviceId}/metrics/{session}` | ✅ | ✅ | ❌ No |
| POST | `/api/devices/{deviceId}/metrics/subscribe` | ✅ | ✅ | ❌ No |
| POST | `/api/devices/{deviceId}/metrics/unsubscribe` | ✅ | ✅ | ❌ No |
| POST | `/api/devices/{deviceId}/metrics/configure` | ✅ | ✅ | ❌ No |
| GET | `/api/chat/messages` | ✅ | ✅ | ❌ No |
| POST | `/api/chat/message` | ✅ | ✅ | ❌ No |
| GET | `/api/schemas` | ✅ | ✅ | ❌ No |
| GET | `/api/schemas/{topic}/{version}/schema.json` | ✅ | ✅ | ❌ No |

### ❌ Endpoints NOT Documented in OpenAPI

| Method | Endpoint | Server Code | Client Usage |
|--------|----------|-------------|--------------|
| GET | `/api/admin/devices` | ✅ | Admin.tsx |
| POST | `/api/admin/devices` | ✅ | Admin.tsx |
| PATCH | `/api/admin/devices/{deviceId}` | ✅ | Admin.tsx (api.ts) |
| DELETE | `/api/admin/devices/{deviceId}` | ✅ | Admin.tsx |
| GET | `/api/admin/services` | ✅ | Admin.tsx |
| POST | `/api/admin/services` | ✅ | Admin.tsx |
| PATCH | `/api/admin/services/{serviceId}` | ❌ Missing | Admin.tsx (api.ts) |
| DELETE | `/api/admin/services/{serviceId}` | ❌ Missing | Admin.tsx |

## Critical Issues

### 1. No Runtime Validation
**Impact**: 🔴 **HIGH SECURITY RISK**

The server does **NOT validate** request bodies or query parameters against any schema. For example:

```typescript
// routes.ts - POST /api/devices
app.post("/api/devices", async (req, res) => {
  const { id, name, type, ip, specs } = req.body; // ❌ No validation!
  await db.insert(devices).values({ id, name, type, ip, specs });
});
```

**Risks**:
- SQL injection via malformed data
- Type mismatches causing crashes
- Missing required fields
- Invalid enum values
- Buffer overflow attacks via oversized strings

### 2. Missing OpenAPI Documentation
**Impact**: 🟡 **MEDIUM**

Admin endpoints (`/api/admin/*`) are not documented in OpenAPI spec but are actively used by the client.

### 3. Incomplete Admin Services Implementation
**Impact**: 🟡 **MEDIUM**

Client expects PATCH/DELETE for services, but server returns 501 (Not Implemented).

### 4. No Response Validation
**Impact**: 🟡 **MEDIUM**

Server responses are not validated to match OpenAPI schemas before sending to clients.

## Browser API Usage Summary

### Authentication (api.ts)
```typescript
auth.login(email, password)          → POST /api/auth/login
auth.oidcLogin(provider)             → POST /api/auth/login
auth.logout()                        → POST /api/auth/logout
```

### Devices (api.ts)
```typescript
devices.list(status?)                → GET /api/devices?status={status}
devices.connect(deviceId)            → POST /api/devices/{deviceId}/connect  ❌ NOT IN SERVER!
devices.disconnect(deviceId)         → POST /api/devices/{deviceId}/disconnect  ❌ NOT IN SERVER!
devices.scan()                       → POST /api/devices/scan  ❌ NOT IN SERVER!
```

### Chat (api.ts)
```typescript
chat.getMessages(deviceId?)          → GET /api/chat/messages?deviceId={id}
chat.sendMessage(msg, images, ...)   → POST /api/chat/message (multipart/form-data)
```

### Admin (api.ts)
```typescript
admin.devices.list()                 → GET /api/admin/devices
admin.devices.create(data)           → POST /api/admin/devices
admin.devices.update(id, data)       → PATCH /api/admin/devices/{id}
admin.devices.delete(id)             → DELETE /api/admin/devices/{id}

admin.services.list()                → GET /api/admin/services
admin.services.create(data)          → POST /api/admin/services
admin.services.update(id, data)      → PATCH /api/admin/services/{id}  ❌ NOT IN SERVER!
admin.services.delete(id)            → DELETE /api/admin/services/{id}  ❌ NOT IN SERVER!
```

### Direct Fetch Calls (DeviceDetails.tsx)
```typescript
fetch(`/api/devices/${id}/snapshot`, { method: 'POST' })
```

## Recommendations

### Priority 1: Add Request Validation Middleware

Install OpenAPI validator:
```bash
cd web-platform
npm install express-openapi-validator
```

Add to routes.ts:
```typescript
import OpenApiValidator from 'express-openapi-validator';

export async function registerRoutes(app: Express) {
  // Add before route definitions
  app.use(
    OpenApiValidator.middleware({
      apiSpec: join(__dirname, '../../shared/specs/openapi.yaml'),
      validateRequests: true,
      validateResponses: true,
      validateSecurity: true,
    })
  );
  
  // ... rest of routes
}
```

### Priority 2: Complete OpenAPI Spec

Add missing admin endpoints to `shared/specs/openapi.yaml`:

```yaml
/api/admin/devices:
  get:
    summary: List all devices (admin)
    tags: [admin]
  post:
    summary: Create device (admin)
    tags: [admin]

/api/admin/devices/{deviceId}:
  patch:
    summary: Update device (admin)
    tags: [admin]
  delete:
    summary: Delete device (admin)
    tags: [admin]

/api/admin/services:
  get:
    summary: List services (admin)
    tags: [admin]
  post:
    summary: Create service (admin)
    tags: [admin]

/api/admin/services/{serviceId}:
  patch:
    summary: Update service (admin)
    tags: [admin]
  delete:
    summary: Delete service (admin)
    tags: [admin]

/api/devices/{deviceId}/connect:
  post:
    summary: Connect to device
    tags: [devices]

/api/devices/{deviceId}/disconnect:
  post:
    summary: Disconnect from device
    tags: [devices]

/api/devices/scan:
  post:
    summary: Scan for devices
    tags: [devices]
```

### Priority 3: Implement Missing Endpoints

Complete services implementation in routes.ts:
```typescript
app.patch("/api/admin/services/:serviceId", async (req, res) => {
  // Implementation
});

app.delete("/api/admin/services/:serviceId", async (req, res) => {
  // Implementation
});
```

Add connect/disconnect/scan endpoints:
```typescript
app.post("/api/devices/:deviceId/connect", async (req, res) => {
  // Implementation
});

app.post("/api/devices/:deviceId/disconnect", async (req, res) => {
  // Implementation
});

app.post("/api/devices/scan", async (req, res) => {
  // Implementation
});
```

### Priority 4: Add Manual Validation (Interim)

If OpenAPI validator integration is complex, add Zod validation:

```typescript
import { z } from 'zod';

const CreateDeviceSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  type: z.enum(['raspberry-pi', 'jetson', 'coral', 'ncs', 'other']),
  ip: z.string().ip(),
  specs: z.object({}).passthrough().optional()
});

app.post("/api/devices", async (req, res) => {
  try {
    const validated = CreateDeviceSchema.parse(req.body);
    const [device] = await db.insert(devices).values(validated).returning();
    res.json(device);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Validation failed', details: error.errors });
    }
    res.status(500).json({ error: 'Failed to create device' });
  }
});
```

## Action Items

- [ ] Install and configure `express-openapi-validator`
- [ ] Add admin endpoints to OpenAPI spec
- [ ] Add device connect/disconnect/scan endpoints to OpenAPI spec
- [ ] Implement missing service PATCH/DELETE endpoints
- [ ] Implement missing device connect/disconnect/scan endpoints
- [ ] Add validation middleware to all routes
- [ ] Add error handler for validation failures
- [ ] Test all endpoints with invalid inputs
- [ ] Update client error handling for 400 validation errors
