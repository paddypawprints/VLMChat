# VLM Status UI Implementation

## Summary

Implemented visual status badges to distinguish VLM-validated detections from timeout/error cases in the web platform UI.

## Components Created

### VlmStatusBadge
**File**: `client/src/components/VlmStatusBadge.tsx`

Displays VLM verification status with color-coded badges:

| Status | Badge Text | Color | Icon | Condition |
|--------|-----------|-------|------|-----------|
| ✅ Verified | "VLM Verified" | Green | CheckCircle2 | `vlmVerified === true` |
| ✅ Direct | "Detected" | Green | CheckCircle2 | `vlmVerified === false && !vlmRequired` |
| ⚠️ Timeout | "VLM Timeout" | Yellow | Clock | `vlmTimeout === true` |
| ⚠️ Error | "VLM Error" | Yellow | AlertTriangle | `vlmError` present |

## Updated Components

### 1. DeviceDetails Page
**File**: `client/src/pages/DeviceDetails.tsx`

**Changes**:
- Added VLM status badge to alert cards (positioned top-right)
- Display VLM result text below description (if available)
- Show VLM error messages in yellow text

**Visual Layout**:
```
Alert Card with VLM Verified:
┌─────────────────────────────────────┐
│ [Image]           95%               │  <- Confidence (top-left)
│                   ✓ VLM Verified    │  <- Status badge (top-right)
├─────────────────────────────────────┤
│ Person in red jacket                │  <- Description
│ 🕒 2:34 PM                          │  <- Timestamp
│ "Wearing red jacket and jeans"      │  <- VLM result (italic)
└─────────────────────────────────────┘

Alert Card with VLM Timeout:
┌─────────────────────────────────────┐
│ [Image]           92%               │
│                   ⚠ Timeout         │  <- Warning badge
├─────────────────────────────────────┤
│ Person detected                     │
│ 🕒 2:35 PM                          │
│ ⚠ VLM queue full (timeout)         │  <- Error message (yellow)
└─────────────────────────────────────┘
```

### 2. DetectionDetailDialog
**File**: `client/src/components/DetectionDetailDialog.tsx`

**Changes**:
- Added VLM fields to Detection interface
- Display VLM status badge in metadata section
- New "VLM Verification" section with result/error details

## Data Schema

Alerts from backend now include VLM fields:

```typescript
interface Alert {
  // Core fields
  type: string;
  timestamp: string;
  confidence: number;
  description: string;
  image?: string;
  
  // VLM verification fields (added by alert_publisher)
  vlm_verified?: boolean;      // True if VLM confirmed
  vlm_required?: boolean;       // True if VLM check was needed
  vlm_timeout?: boolean;        // True if VLM queue was full
  vlm_error?: string;           // Error message if VLM failed
  vlm_result?: string;          // VLM-generated description
  vlm_inference_time?: number;  // Time taken (ms)
}
```

## Status Logic

**Green (Confident)**:
- VLM successfully verified the detection
- OR detection didn't require VLM verification

**Yellow (Unsure)**:
- VLM timed out (queue was full)
- OR VLM encountered an error during processing

## Visual Design

**Colors**:
- Green: `bg-green-100 text-green-800 border-green-300`
- Yellow: `bg-yellow-100 text-yellow-800 border-yellow-300`
- Dark mode variants included

**Icons** (lucide-react):
- ✓ CheckCircle2 - Verified/successful states
- ⏱ Clock - Timeout state
- ⚠ AlertTriangle - Error state

## Testing

1. Start web platform: `cd web-platform && npm run dev`
2. Navigate to device details page
3. Trigger alerts with different VLM states:
   - Direct detection: Set `vlm_required=False` in filter
   - VLM verified: Detection passes through SmolVLM worker
   - VLM timeout: Fill VLM queue (max_queue_size detections)
   - VLM error: Cause SmolVLM processing error

## Files Modified

1. **Created**: `client/src/components/VlmStatusBadge.tsx` (73 lines)
2. **Updated**: `client/src/pages/DeviceDetails.tsx` (added VlmStatusBadge import + alert card badges)
3. **Updated**: `client/src/components/DetectionDetailDialog.tsx` (added VLM fields + verification section)

## Backend Integration

VLM fields are already populated by:
- `macos-device/macos_device/vlm_queue.py` - Sets `vlm_verified=False` for direct routes/timeouts
- `macos-device/macos_device/smolvlm_worker.py` - Sets `vlm_verified=True/False` after VLM processing
- `macos-device/macos_device/alert_publisher.py` - Includes all VLM fields in MQTT alert payload

No backend changes needed - UI consumes existing fields.
