
    const schema = {
  "asyncapi": "3.0.0",
  "info": {
    "title": "Edge AI Platform MQTT API",
    "version": "1.0.0",
    "description": "MQTT messaging protocol for Edge AI Platform devices.\nDefines topics for device registration, status updates, alerts, metrics, and WebRTC signaling.\n",
    "contact": {
      "name": "VLMChat Platform"
    },
    "license": {
      "name": "MIT"
    }
  },
  "servers": {
    "production": {
      "host": "localhost:1883",
      "protocol": "mqtt",
      "description": "Local MQTT broker",
      "tags": [
        {
          "name": "env:development"
        }
      ]
    },
    "docker": {
      "host": "mqtt:1883",
      "protocol": "mqtt",
      "description": "Docker MQTT broker",
      "tags": [
        {
          "name": "env:docker"
        }
      ]
    }
  },
  "defaultContentType": "application/json",
  "channels": {
    "deviceRegister": {
      "address": "devices/{deviceId}/register",
      "messages": {
        "deviceRegister": {
          "name": "DeviceRegister",
          "title": "Device Registration",
          "summary": "Device registration information",
          "contentType": "application/json",
          "payload": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$comment": "⚠️  SECURITY CRITICAL - DO NOT MODIFY WITHOUT REVIEW. This schema validates incoming MQTT messages and protects against malformed or malicious data. Changes must be coordinated across all validation layers (server MQTT validator, device validator, API specs).",
            "$id": "register-v1.0.0.json",
            "title": "Device Registration",
            "description": "Device announces itself and its capabilities to the server",
            "type": "object",
            "required": [
              "device_id",
              "type",
              "capabilities",
              "schema_versions",
              "jwt"
            ],
            "properties": {
              "device_id": {
                "type": "string",
                "description": "Unique device identifier (UUID from manufacturing)",
                "x-parser-schema-id": "<anonymous-schema-2>"
              },
              "jwt": {
                "type": "string",
                "description": "Signed JWT token for device authentication (dev: in payload, prod: in MQTT credentials)",
                "x-parser-schema-id": "<anonymous-schema-3>"
              },
              "type": {
                "type": "string",
                "enum": [
                  "raspberry-pi",
                  "jetson",
                  "coral",
                  "ncs",
                  "other"
                ],
                "description": "Device hardware type",
                "x-parser-schema-id": "<anonymous-schema-4>"
              },
              "name": {
                "type": "string",
                "description": "Human-readable device name",
                "x-parser-schema-id": "<anonymous-schema-5>"
              },
              "ip": {
                "type": "string",
                "format": "ipv4",
                "description": "Device IP address",
                "x-parser-schema-id": "<anonymous-schema-6>"
              },
              "capabilities": {
                "type": "array",
                "items": {
                  "type": "string",
                  "enum": [
                    "ai-detection",
                    "metrics",
                    "webrtc",
                    "camera"
                  ],
                  "x-parser-schema-id": "<anonymous-schema-8>"
                },
                "description": "Device capabilities",
                "x-parser-schema-id": "<anonymous-schema-7>"
              },
              "instruments": {
                "type": "array",
                "items": {
                  "type": "object",
                  "required": [
                    "name",
                    "type"
                  ],
                  "properties": {
                    "name": {
                      "type": "string",
                      "description": "Instrument name",
                      "x-parser-schema-id": "<anonymous-schema-11>"
                    },
                    "type": {
                      "type": "string",
                      "description": "Instrument class type",
                      "x-parser-schema-id": "<anonymous-schema-12>"
                    }
                  },
                  "x-parser-schema-id": "<anonymous-schema-10>"
                },
                "description": "Available metrics instruments on device",
                "x-parser-schema-id": "<anonymous-schema-9>"
              },
              "sessions": {
                "type": "array",
                "items": {
                  "type": "string",
                  "x-parser-schema-id": "<anonymous-schema-14>"
                },
                "description": "Available metrics session names",
                "x-parser-schema-id": "<anonymous-schema-13>"
              },
              "schema_versions": {
                "type": "object",
                "description": "Schema versions supported by this device",
                "properties": {
                  "device-register": {
                    "type": "string",
                    "pattern": "^v\\d+\\.\\d+\\.\\d+$",
                    "x-parser-schema-id": "<anonymous-schema-16>"
                  },
                  "watchlist": {
                    "type": "string",
                    "pattern": "^v\\d+\\.\\d+\\.\\d+$",
                    "x-parser-schema-id": "<anonymous-schema-17>"
                  },
                  "alerts": {
                    "type": "string",
                    "pattern": "^v\\d+\\.\\d+\\.\\d+$",
                    "x-parser-schema-id": "<anonymous-schema-18>"
                  },
                  "metrics-config": {
                    "type": "string",
                    "pattern": "^v\\d+\\.\\d+\\.\\d+$",
                    "x-parser-schema-id": "<anonymous-schema-19>"
                  },
                  "metrics-data": {
                    "type": "string",
                    "pattern": "^v\\d+\\.\\d+\\.\\d+$",
                    "x-parser-schema-id": "<anonymous-schema-20>"
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-15>"
              },
              "specs": {
                "type": "object",
                "description": "Hardware specifications",
                "properties": {
                  "cpu": {
                    "type": "string",
                    "x-parser-schema-id": "<anonymous-schema-22>"
                  },
                  "memory": {
                    "type": "string",
                    "x-parser-schema-id": "<anonymous-schema-23>"
                  },
                  "temperature": {
                    "type": "number",
                    "x-parser-schema-id": "<anonymous-schema-24>"
                  },
                  "usage": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "x-parser-schema-id": "<anonymous-schema-25>"
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-21>"
              }
            }
          },
          "x-parser-unique-object-id": "deviceRegister"
        }
      },
      "description": "Device sends registration information on startup",
      "parameters": {
        "deviceId": {
          "description": "Unique device identifier",
          "location": "$message.payload#/deviceId"
        }
      },
      "x-parser-unique-object-id": "deviceRegister"
    },
    "deviceStatus": {
      "address": "devices/{deviceId}/status",
      "messages": {
        "deviceStatus": {
          "name": "DeviceStatus",
          "title": "Device Status",
          "summary": "Device status update",
          "contentType": "application/json",
          "payload": {
            "type": "object",
            "required": [
              "status",
              "timestamp"
            ],
            "properties": {
              "status": {
                "type": "string",
                "enum": [
                  "connected",
                  "disconnected",
                  "connecting",
                  "error"
                ],
                "x-parser-schema-id": "<anonymous-schema-28>"
              },
              "timestamp": {
                "type": "string",
                "format": "date-time",
                "x-parser-schema-id": "<anonymous-schema-29>"
              },
              "error": {
                "type": "string",
                "description": "Error message if status is error",
                "x-parser-schema-id": "<anonymous-schema-30>"
              }
            },
            "x-parser-schema-id": "<anonymous-schema-27>"
          },
          "x-parser-unique-object-id": "deviceStatus"
        }
      },
      "description": "Device publishes status changes",
      "parameters": {
        "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
      },
      "x-parser-unique-object-id": "deviceStatus"
    },
    "deviceHeartbeat": {
      "address": "devices/{deviceId}/heartbeat",
      "messages": {
        "deviceHeartbeat": {
          "name": "DeviceHeartbeat",
          "title": "Device Heartbeat",
          "summary": "Periodic keepalive",
          "contentType": "application/json",
          "payload": {
            "type": "object",
            "required": [
              "timestamp"
            ],
            "properties": {
              "timestamp": {
                "type": "string",
                "format": "date-time",
                "x-parser-schema-id": "<anonymous-schema-33>"
              },
              "uptime": {
                "type": "number",
                "description": "Device uptime in seconds",
                "x-parser-schema-id": "<anonymous-schema-34>"
              }
            },
            "x-parser-schema-id": "<anonymous-schema-32>"
          },
          "x-parser-unique-object-id": "deviceHeartbeat"
        }
      },
      "description": "Periodic heartbeat from device (keepalive)",
      "parameters": {
        "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
      },
      "x-parser-unique-object-id": "deviceHeartbeat"
    },
    "deviceAlerts": {
      "address": "devices/{deviceId}/alerts",
      "messages": {
        "alert": {
          "name": "Alert",
          "title": "Detection Alert",
          "summary": "Detection alert or system event",
          "contentType": "application/json",
          "payload": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$comment": "⚠️  SECURITY CRITICAL - DO NOT MODIFY WITHOUT REVIEW. This schema validates incoming MQTT messages and protects against malformed or malicious data. Changes must be coordinated across all validation layers (server MQTT validator, device validator, API specs).",
            "$id": "alerts-v1.0.0.json",
            "title": "Alert",
            "description": "Detection alert or system event from device",
            "type": "object",
            "required": [
              "type",
              "timestamp"
            ],
            "properties": {
              "type": {
                "type": "string",
                "enum": [
                  "detection",
                  "system",
                  "error"
                ],
                "description": "Alert type",
                "x-parser-schema-id": "<anonymous-schema-36>"
              },
              "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "When the alert occurred",
                "x-parser-schema-id": "<anonymous-schema-37>"
              },
              "watchlist_item_id": {
                "type": "string",
                "description": "ID of matched watchlist item (for detection alerts)",
                "x-parser-schema-id": "<anonymous-schema-38>"
              },
              "description": {
                "type": "string",
                "description": "Alert description",
                "x-parser-schema-id": "<anonymous-schema-39>"
              },
              "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Detection confidence score",
                "x-parser-schema-id": "<anonymous-schema-40>"
              },
              "image": {
                "type": "string",
                "description": "Base64 encoded image or reference",
                "x-parser-schema-id": "<anonymous-schema-41>"
              },
              "image_url": {
                "type": "string",
                "format": "uri",
                "description": "URL to image in storage",
                "x-parser-schema-id": "<anonymous-schema-42>"
              },
              "metadata": {
                "type": "object",
                "description": "Additional alert metadata",
                "properties": {
                  "bounding_box": {
                    "type": "object",
                    "properties": {
                      "x": {
                        "type": "number",
                        "x-parser-schema-id": "<anonymous-schema-45>"
                      },
                      "y": {
                        "type": "number",
                        "x-parser-schema-id": "<anonymous-schema-46>"
                      },
                      "width": {
                        "type": "number",
                        "x-parser-schema-id": "<anonymous-schema-47>"
                      },
                      "height": {
                        "type": "number",
                        "x-parser-schema-id": "<anonymous-schema-48>"
                      }
                    },
                    "x-parser-schema-id": "<anonymous-schema-44>"
                  },
                  "inference_time_ms": {
                    "type": "number",
                    "description": "Time taken for inference",
                    "x-parser-schema-id": "<anonymous-schema-49>"
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-43>"
              }
            }
          },
          "x-parser-unique-object-id": "alert"
        }
      },
      "description": "Device publishes detection alerts and system events",
      "parameters": {
        "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
      },
      "x-parser-unique-object-id": "deviceAlerts"
    },
    "deviceSnapshot": {
      "address": "devices/{deviceId}/snapshot",
      "messages": {
        "snapshot": {
          "name": "Snapshot",
          "title": "Camera Snapshot",
          "summary": "Camera image snapshot with optional object detections",
          "contentType": "application/json",
          "payload": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$comment": "⚠️  SECURITY CRITICAL - DO NOT MODIFY WITHOUT REVIEW. This schema validates incoming MQTT messages and protects against malformed or malicious data. Changes must be coordinated across all validation layers (server MQTT validator, device validator, API specs).",
            "$id": "snapshot-v1.0.0.json",
            "title": "Device Snapshot",
            "description": "Schema for device snapshot messages containing captured images with optional object detections",
            "type": "object",
            "required": [
              "device_id",
              "timestamp",
              "image"
            ],
            "properties": {
              "device_id": {
                "type": "string",
                "description": "Unique identifier for the device",
                "minLength": 1,
                "x-parser-schema-id": "<anonymous-schema-51>"
              },
              "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "ISO 8601 UTC timestamp when the snapshot was captured",
                "x-parser-schema-id": "<anonymous-schema-52>"
              },
              "image": {
                "type": "string",
                "description": "Base64-encoded image data",
                "minLength": 1,
                "x-parser-schema-id": "<anonymous-schema-53>"
              },
              "format": {
                "type": "string",
                "enum": [
                  "jpeg",
                  "png"
                ],
                "description": "Image format",
                "default": "jpeg",
                "x-parser-schema-id": "<anonymous-schema-54>"
              },
              "width": {
                "type": "integer",
                "description": "Image width in pixels",
                "minimum": 1,
                "x-parser-schema-id": "<anonymous-schema-55>"
              },
              "height": {
                "type": "integer",
                "description": "Image height in pixels",
                "minimum": 1,
                "x-parser-schema-id": "<anonymous-schema-56>"
              },
              "detections": {
                "type": "array",
                "description": "Array of object detections from YOLO or other vision models (tree structure with children)",
                "items": {
                  "$schema": "http://json-schema.org/draft-07/schema#",
                  "$id": "detection-v1.0.0.json",
                  "title": "Object Detection",
                  "description": "Schema for object detection with recursive children support (for clustered detections)",
                  "type": "object",
                  "required": [
                    "bbox",
                    "confidence",
                    "category"
                  ],
                  "properties": {
                    "bbox": {
                      "type": "array",
                      "description": "Bounding box coordinates [x1, y1, x2, y2] in pixels",
                      "items": {
                        "type": "number",
                        "x-parser-schema-id": "<anonymous-schema-59>"
                      },
                      "minItems": 4,
                      "maxItems": 4,
                      "x-parser-schema-id": "<anonymous-schema-58>"
                    },
                    "confidence": {
                      "type": "number",
                      "description": "Detection confidence score (0.0 to 1.0)",
                      "minimum": 0,
                      "maximum": 1,
                      "x-parser-schema-id": "<anonymous-schema-60>"
                    },
                    "category": {
                      "type": "string",
                      "description": "Object category label (e.g., 'person', 'car', 'dog')",
                      "minLength": 1,
                      "x-parser-schema-id": "<anonymous-schema-61>"
                    },
                    "category_id": {
                      "type": "integer",
                      "description": "COCO category ID (-1 for unknown)",
                      "x-parser-schema-id": "<anonymous-schema-62>"
                    },
                    "children": {
                      "type": "array",
                      "description": "Child detections (for clustered objects)",
                      "items": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "$id": "detection-v1.0.0.json",
                        "title": "Object Detection",
                        "description": "Schema for object detection with recursive children support (for clustered detections)",
                        "type": "object",
                        "required": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.required",
                        "properties": {
                          "bbox": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.bbox",
                          "confidence": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.confidence",
                          "category": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.category",
                          "category_id": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.category_id",
                          "children": {
                            "type": "array",
                            "description": "Child detections (for clustered objects)",
                            "items": {
                              "asyncapi": "3.0.0",
                              "info": "$ref:$.info",
                              "servers": "$ref:$.servers",
                              "defaultContentType": "application/json",
                              "channels": "$ref:$.channels",
                              "operations": {
                                "publishDeviceRegister": {
                                  "action": "send",
                                  "channel": "$ref:$.channels.deviceRegister",
                                  "summary": "Device publishes registration",
                                  "description": "Sent when device starts up or reconnects",
                                  "x-parser-unique-object-id": "publishDeviceRegister"
                                },
                                "publishDeviceStatus": {
                                  "action": "send",
                                  "channel": "$ref:$.channels.deviceStatus",
                                  "summary": "Device publishes status update",
                                  "description": "Sent when device status changes",
                                  "x-parser-unique-object-id": "publishDeviceStatus"
                                },
                                "publishDeviceHeartbeat": {
                                  "action": "send",
                                  "channel": "$ref:$.channels.deviceHeartbeat",
                                  "summary": "Device sends heartbeat",
                                  "description": "Periodic keepalive message",
                                  "x-parser-unique-object-id": "publishDeviceHeartbeat"
                                },
                                "publishAlert": {
                                  "action": "send",
                                  "channel": "$ref:$.channels.deviceAlerts",
                                  "summary": "Device publishes alert",
                                  "description": "Detection alert or system event",
                                  "x-parser-unique-object-id": "publishAlert"
                                },
                                "sendFilterCommand": {
                                  "action": "send",
                                  "channel": {
                                    "address": "devices/{deviceId}/commands/filter",
                                    "messages": {
                                      "filterCommand": {
                                        "name": "FilterCommand",
                                        "title": "Filter Configuration Command",
                                        "summary": "Update detection filter configuration on device",
                                        "contentType": "application/json",
                                        "payload": {
                                          "$schema": "http://json-schema.org/draft-07/schema#",
                                          "$id": "filter-config-v1.0.0.json",
                                          "title": "Filter Configuration",
                                          "description": "Detection filter configuration using boolean vectors and color matching for categories and attributes",
                                          "type": "object",
                                          "required": [
                                            "category_mask",
                                            "category_colors",
                                            "attribute_mask",
                                            "attribute_colors"
                                          ],
                                          "properties": {
                                            "category_mask": {
                                              "type": "array",
                                              "description": "Boolean vector for COCO categories (80 items, indexed by category.id)",
                                              "items": {
                                                "type": "boolean",
                                                "x-parser-schema-id": "<anonymous-schema-83>"
                                              },
                                              "minItems": 80,
                                              "maxItems": 80,
                                              "x-parser-schema-id": "<anonymous-schema-82>"
                                            },
                                            "category_colors": {
                                              "type": "array",
                                              "description": "Color matching for each COCO category (80 items, null if no color filter, hex string if filtering by color)",
                                              "items": {
                                                "oneOf": [
                                                  {
                                                    "type": "null",
                                                    "x-parser-schema-id": "<anonymous-schema-86>"
                                                  },
                                                  {
                                                    "type": "string",
                                                    "pattern": "^#[0-9A-Fa-f]{6}$",
                                                    "description": "Hex color code (e.g., '#ff0000' for red)",
                                                    "x-parser-schema-id": "<anonymous-schema-87>"
                                                  }
                                                ],
                                                "x-parser-schema-id": "<anonymous-schema-85>"
                                              },
                                              "minItems": 80,
                                              "maxItems": 80,
                                              "x-parser-schema-id": "<anonymous-schema-84>"
                                            },
                                            "attribute_mask": {
                                              "type": "array",
                                              "description": "Boolean vector for PA100K attributes (26 items, same order as PA100K.ATTRIBUTES)",
                                              "items": {
                                                "type": "boolean",
                                                "x-parser-schema-id": "<anonymous-schema-89>"
                                              },
                                              "minItems": 26,
                                              "maxItems": 26,
                                              "x-parser-schema-id": "<anonymous-schema-88>"
                                            },
                                            "attribute_colors": {
                                              "type": "array",
                                              "description": "Color matching for each PA100K attribute (26 items, null if no color filter, hex string if filtering by color)",
                                              "items": {
                                                "oneOf": [
                                                  {
                                                    "type": "null",
                                                    "x-parser-schema-id": "<anonymous-schema-92>"
                                                  },
                                                  {
                                                    "type": "string",
                                                    "pattern": "^#[0-9A-Fa-f]{6}$",
                                                    "description": "Hex color code (e.g., '#0000ff' for blue)",
                                                    "x-parser-schema-id": "<anonymous-schema-93>"
                                                  }
                                                ],
                                                "x-parser-schema-id": "<anonymous-schema-91>"
                                              },
                                              "minItems": 26,
                                              "maxItems": 26,
                                              "x-parser-schema-id": "<anonymous-schema-90>"
                                            }
                                          },
                                          "additionalProperties": false,
                                          "examples": [
                                            {
                                              "category_mask": [
                                                true,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false
                                              ],
                                              "category_colors": [
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null
                                              ],
                                              "attribute_mask": [
                                                true,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                true,
                                                false,
                                                false,
                                                false,
                                                true,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false
                                              ],
                                              "attribute_colors": [
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                "#00ff00",
                                                null,
                                                null,
                                                null,
                                                "#ff0000",
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null,
                                                null
                                              ]
                                            }
                                          ]
                                        },
                                        "x-parser-unique-object-id": "filterCommand"
                                      }
                                    },
                                    "description": "Server sends detection filter configuration to device.\nUpdates category_mask (80 bools for COCO categories) and\nattribute_mask (26 bools for PA100K attributes).\n",
                                    "parameters": {
                                      "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
                                    },
                                    "x-parser-unique-object-id": "filterCommand"
                                  },
                                  "summary": "Platform sends filter configuration",
                                  "description": "Update device detection filter with complete filter list",
                                  "x-parser-unique-object-id": "sendFilterCommand"
                                },
                                "sendConfigCommand": {
                                  "action": "send",
                                  "channel": {
                                    "address": "devices/{deviceId}/commands/config",
                                    "messages": {
                                      "configCommand": {
                                        "name": "ConfigCommand",
                                        "title": "Device Configuration Command",
                                        "summary": "Update device configuration (tasks and sinks)",
                                        "contentType": "application/json",
                                        "payload": {
                                          "type": "object",
                                          "required": [
                                            "version",
                                            "config"
                                          ],
                                          "properties": {
                                            "version": {
                                              "type": "integer",
                                              "description": "Configuration version number (for tracking)",
                                              "minimum": 1,
                                              "x-parser-schema-id": "<anonymous-schema-96>"
                                            },
                                            "config": {
                                              "type": "object",
                                              "description": "Full device configuration",
                                              "required": [
                                                "tasks",
                                                "sinks"
                                              ],
                                              "properties": {
                                                "tasks": {
                                                  "type": "object",
                                                  "description": "Task configurations (YOLO, attributes, clusterer, tracker, etc.)",
                                                  "additionalProperties": true,
                                                  "x-parser-schema-id": "<anonymous-schema-98>"
                                                },
                                                "sinks": {
                                                  "type": "object",
                                                  "description": "Sink configurations (MQTT, storage, etc.)",
                                                  "additionalProperties": true,
                                                  "x-parser-schema-id": "<anonymous-schema-99>"
                                                }
                                              },
                                              "x-parser-schema-id": "<anonymous-schema-97>"
                                            }
                                          },
                                          "x-parser-schema-id": "<anonymous-schema-95>"
                                        },
                                        "x-parser-unique-object-id": "configCommand"
                                      }
                                    },
                                    "description": "Server sends device configuration updates.\nContains full device config (tasks, sinks) with version number.\nDevice should validate, apply, and respond with status.\n",
                                    "parameters": {
                                      "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
                                    },
                                    "x-parser-unique-object-id": "configCommand"
                                  },
                                  "summary": "Platform sends device configuration",
                                  "description": "Update device configuration with new tasks and sinks settings.\nDevice should validate, apply, and respond with status update.\n",
                                  "x-parser-unique-object-id": "sendConfigCommand"
                                },
                                "publishSnapshot": {
                                  "action": "send",
                                  "channel": "$ref:$.channels.deviceSnapshot",
                                  "summary": "Device publishes snapshot",
                                  "description": "Camera snapshot image",
                                  "x-parser-unique-object-id": "publishSnapshot"
                                },
                                "sendMetricsCommand": {
                                  "action": "send",
                                  "channel": {
                                    "address": "devices/{deviceId}/commands/metrics",
                                    "messages": {
                                      "metricsCommand": {
                                        "name": "MetricsCommand",
                                        "title": "Metrics Command",
                                        "summary": "Control and configure metrics collection on device",
                                        "contentType": "application/json",
                                        "payload": {
                                          "type": "object",
                                          "required": [
                                            "enabled"
                                          ],
                                          "properties": {
                                            "enabled": {
                                              "type": "boolean",
                                              "description": "Enable (true) or disable (false) metrics collection",
                                              "x-parser-schema-id": "<anonymous-schema-75>"
                                            },
                                            "frequency": {
                                              "type": "number",
                                              "description": "Publishing interval in seconds",
                                              "default": 30,
                                              "minimum": 1,
                                              "x-parser-schema-id": "<anonymous-schema-76>"
                                            },
                                            "instruments": {
                                              "description": "Array of instrument names to collect, or \"*\" for all",
                                              "oneOf": [
                                                {
                                                  "type": "string",
                                                  "enum": [
                                                    "*"
                                                  ],
                                                  "x-parser-schema-id": "<anonymous-schema-78>"
                                                },
                                                {
                                                  "type": "array",
                                                  "items": {
                                                    "type": "string",
                                                    "x-parser-schema-id": "<anonymous-schema-80>"
                                                  },
                                                  "x-parser-schema-id": "<anonymous-schema-79>"
                                                }
                                              ],
                                              "x-parser-schema-id": "<anonymous-schema-77>"
                                            }
                                          },
                                          "x-parser-schema-id": "<anonymous-schema-74>"
                                        },
                                        "x-parser-unique-object-id": "metricsCommand"
                                      }
                                    },
                                    "description": "Server sends metrics collection commands to device.\nUse enabled=true to start metrics, enabled=false to stop.\nOptional frequency and instruments parameters configure collection.\n",
                                    "parameters": {
                                      "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
                                    },
                                    "x-parser-unique-object-id": "metricsCommand"
                                  },
                                  "summary": "Server controls metrics collection",
                                  "description": "Start metrics with enabled=true, stop with enabled=false.\nConfigure frequency and instruments as needed.\n",
                                  "x-parser-unique-object-id": "sendMetricsCommand"
                                },
                                "publishMetricsData": {
                                  "action": "send",
                                  "channel": {
                                    "address": "devices/{deviceId}/metrics",
                                    "messages": {
                                      "metricsData": {
                                        "name": "MetricsData",
                                        "title": "Metrics Data",
                                        "summary": "Instrumentation values from metrics session",
                                        "contentType": "application/json",
                                        "payload": {
                                          "type": "object",
                                          "required": [
                                            "timestamp",
                                            "session"
                                          ],
                                          "properties": {
                                            "timestamp": {
                                              "type": "string",
                                              "format": "date-time",
                                              "description": "Data snapshot timestamp",
                                              "x-parser-schema-id": "<anonymous-schema-102>"
                                            },
                                            "session": {
                                              "type": "object",
                                              "description": "Current metrics session state",
                                              "properties": {
                                                "start_time": {
                                                  "type": "number",
                                                  "description": "Session start time (Unix timestamp)",
                                                  "x-parser-schema-id": "<anonymous-schema-104>"
                                                },
                                                "end_time": {
                                                  "type": "number",
                                                  "nullable": true,
                                                  "description": "Session end time (Unix timestamp), null if running",
                                                  "x-parser-schema-id": "<anonymous-schema-105>"
                                                },
                                                "instruments": {
                                                  "type": "array",
                                                  "description": "Array of instrument states",
                                                  "items": {
                                                    "type": "object",
                                                    "required": [
                                                      "timeseries",
                                                      "instrument"
                                                    ],
                                                    "properties": {
                                                      "timeseries": {
                                                        "type": "string",
                                                        "description": "Timeseries name",
                                                        "x-parser-schema-id": "<anonymous-schema-108>"
                                                      },
                                                      "instrument": {
                                                        "type": "object",
                                                        "description": "Instrument state export",
                                                        "required": [
                                                          "type",
                                                          "name"
                                                        ],
                                                        "properties": {
                                                          "type": {
                                                            "type": "string",
                                                            "description": "Instrument type class name",
                                                            "x-parser-schema-id": "<anonymous-schema-110>"
                                                          },
                                                          "name": {
                                                            "type": "string",
                                                            "description": "Instrument name",
                                                            "x-parser-schema-id": "<anonymous-schema-111>"
                                                          }
                                                        },
                                                        "x-parser-schema-id": "<anonymous-schema-109>"
                                                      }
                                                    },
                                                    "x-parser-schema-id": "<anonymous-schema-107>"
                                                  },
                                                  "x-parser-schema-id": "<anonymous-schema-106>"
                                                }
                                              },
                                              "x-parser-schema-id": "<anonymous-schema-103>"
                                            }
                                          },
                                          "x-parser-schema-id": "<anonymous-schema-101>"
                                        },
                                        "x-parser-unique-object-id": "metricsData"
                                      }
                                    },
                                    "description": "Device publishes metrics data at configured frequency",
                                    "parameters": {
                                      "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
                                    },
                                    "x-parser-unique-object-id": "metricsData"
                                  },
                                  "summary": "Device publishes metrics data",
                                  "description": "Instrumentation values at configured interval",
                                  "x-parser-unique-object-id": "publishMetricsData"
                                },
                                "sendLogCommand": {
                                  "action": "send",
                                  "channel": {
                                    "address": "devices/{deviceId}/commands/logs",
                                    "messages": {
                                      "logCommand": {
                                        "name": "LogCommand",
                                        "title": "Log Command",
                                        "summary": "Control log streaming from device",
                                        "contentType": "application/json",
                                        "payload": {
                                          "$schema": "http://json-schema.org/draft-07/schema#",
                                          "$id": "log-command-v1.0.0.json",
                                          "title": "Log Command",
                                          "description": "Command to control device log streaming",
                                          "type": "object",
                                          "properties": {
                                            "enabled": {
                                              "type": "boolean",
                                              "description": "Enable or disable log streaming",
                                              "x-parser-schema-id": "<anonymous-schema-113>"
                                            },
                                            "level": {
                                              "type": "string",
                                              "enum": [
                                                "DEBUG",
                                                "INFO",
                                                "WARNING",
                                                "ERROR",
                                                "CRITICAL"
                                              ],
                                              "description": "Minimum log level to stream (only logs at this level or higher will be sent)",
                                              "x-parser-schema-id": "<anonymous-schema-114>"
                                            }
                                          },
                                          "required": [
                                            "enabled"
                                          ],
                                          "additionalProperties": false
                                        },
                                        "x-parser-unique-object-id": "logCommand"
                                      }
                                    },
                                    "description": "Server sends log streaming commands to device.\nUse enabled=true with level to start log streaming at specified level.\nUse enabled=false to stop log streaming.\n",
                                    "parameters": {
                                      "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
                                    },
                                    "x-parser-unique-object-id": "logCommand"
                                  },
                                  "summary": "Server controls log streaming",
                                  "description": "Start log streaming with enabled=true and level (DEBUG/INFO/WARNING/ERROR/CRITICAL).\nStop with enabled=false. Logs are filtered at source by level.\n",
                                  "x-parser-unique-object-id": "sendLogCommand"
                                },
                                "publishLogEntry": {
                                  "action": "send",
                                  "channel": {
                                    "address": "devices/{deviceId}/logs",
                                    "messages": {
                                      "logEntry": {
                                        "name": "LogEntry",
                                        "title": "Log Entry",
                                        "summary": "Single log entry from device",
                                        "contentType": "application/json",
                                        "payload": {
                                          "$schema": "http://json-schema.org/draft-07/schema#",
                                          "$id": "log-entry-v1.0.0.json",
                                          "title": "Log Entry",
                                          "description": "A single log entry from a device",
                                          "type": "object",
                                          "properties": {
                                            "timestamp": {
                                              "type": "string",
                                              "format": "date-time",
                                              "description": "ISO 8601 timestamp of the log entry",
                                              "x-parser-schema-id": "<anonymous-schema-116>"
                                            },
                                            "level": {
                                              "type": "string",
                                              "enum": [
                                                "DEBUG",
                                                "INFO",
                                                "WARNING",
                                                "ERROR",
                                                "CRITICAL"
                                              ],
                                              "description": "Log level",
                                              "x-parser-schema-id": "<anonymous-schema-117>"
                                            },
                                            "logger": {
                                              "type": "string",
                                              "description": "Logger name (e.g., module or component name)",
                                              "x-parser-schema-id": "<anonymous-schema-118>"
                                            },
                                            "message": {
                                              "type": "string",
                                              "description": "Log message",
                                              "x-parser-schema-id": "<anonymous-schema-119>"
                                            },
                                            "module": {
                                              "type": "string",
                                              "description": "Module or file where log originated",
                                              "x-parser-schema-id": "<anonymous-schema-120>"
                                            },
                                            "line": {
                                              "type": "integer",
                                              "description": "Line number in source file",
                                              "x-parser-schema-id": "<anonymous-schema-121>"
                                            },
                                            "thread": {
                                              "type": "string",
                                              "description": "Thread name or ID",
                                              "x-parser-schema-id": "<anonymous-schema-122>"
                                            }
                                          },
                                          "required": [
                                            "timestamp",
                                            "level",
                                            "logger",
                                            "message"
                                          ],
                                          "additionalProperties": false
                                        },
                                        "x-parser-unique-object-id": "logEntry"
                                      }
                                    },
                                    "description": "Device publishes log entries when log streaming is enabled",
                                    "parameters": {
                                      "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
                                    },
                                    "x-parser-unique-object-id": "deviceLogs"
                                  },
                                  "summary": "Device publishes log entries",
                                  "description": "Log entries streamed when enabled",
                                  "x-parser-unique-object-id": "publishLogEntry"
                                }
                              },
                              "components": {
                                "parameters": {
                                  "DeviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
                                },
                                "messages": {
                                  "DeviceRegister": "$ref:$.channels.deviceRegister.messages.deviceRegister",
                                  "DeviceStatus": "$ref:$.channels.deviceStatus.messages.deviceStatus",
                                  "DeviceHeartbeat": "$ref:$.channels.deviceHeartbeat.messages.deviceHeartbeat",
                                  "Alert": "$ref:$.channels.deviceAlerts.messages.alert",
                                  "Snapshot": "$ref:$.channels.deviceSnapshot.messages.snapshot",
                                  "MetricsCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendMetricsCommand.channel.messages.metricsCommand",
                                  "FilterCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendFilterCommand.channel.messages.filterCommand",
                                  "ConfigCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendConfigCommand.channel.messages.configCommand",
                                  "MetricsData": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.publishMetricsData.channel.messages.metricsData",
                                  "LogCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendLogCommand.channel.messages.logCommand",
                                  "LogEntry": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.publishLogEntry.channel.messages.logEntry",
                                  "WebRTCSignal": {
                                    "name": "WebRTCSignal",
                                    "title": "WebRTC Signaling",
                                    "summary": "WebRTC signaling message",
                                    "contentType": "application/json",
                                    "payload": {
                                      "$schema": "http://json-schema.org/draft-07/schema#",
                                      "$comment": "⚠️  SECURITY CRITICAL - DO NOT MODIFY WITHOUT REVIEW. This schema validates incoming MQTT messages and protects against malformed or malicious data. Changes must be coordinated across all validation layers (server MQTT validator, device validator, API specs).",
                                      "$id": "webrtc-signaling-v1.0.0.json",
                                      "title": "WebRTC Signaling",
                                      "description": "WebRTC offer/answer/ICE signaling messages",
                                      "type": "object",
                                      "required": [
                                        "type"
                                      ],
                                      "properties": {
                                        "type": {
                                          "type": "string",
                                          "enum": [
                                            "offer",
                                            "answer",
                                            "ice"
                                          ],
                                          "description": "Signaling message type",
                                          "x-parser-schema-id": "<anonymous-schema-124>"
                                        },
                                        "sdp": {
                                          "type": "string",
                                          "description": "Session Description Protocol (for offer/answer)",
                                          "x-parser-schema-id": "<anonymous-schema-125>"
                                        },
                                        "candidate": {
                                          "type": "object",
                                          "description": "ICE candidate (for ice type)",
                                          "properties": {
                                            "candidate": {
                                              "type": "string",
                                              "x-parser-schema-id": "<anonymous-schema-127>"
                                            },
                                            "sdpMid": {
                                              "type": "string",
                                              "x-parser-schema-id": "<anonymous-schema-128>"
                                            },
                                            "sdpMLineIndex": {
                                              "type": "number",
                                              "x-parser-schema-id": "<anonymous-schema-129>"
                                            }
                                          },
                                          "x-parser-schema-id": "<anonymous-schema-126>"
                                        },
                                        "request_id": {
                                          "type": "string",
                                          "description": "Request identifier for matching responses",
                                          "x-parser-schema-id": "<anonymous-schema-130>"
                                        }
                                      }
                                    },
                                    "x-parser-unique-object-id": "webrtcSignal"
                                  }
                                }
                              },
                              "x-parser-spec-parsed": true,
                              "x-parser-api-version": 3,
                              "x-parser-circular": true,
                              "x-parser-schema-id": "<anonymous-schema-65>"
                            },
                            "x-parser-schema-id": "<anonymous-schema-64>"
                          },
                          "metadata": {
                            "type": "object",
                            "description": "Extensible metadata for task enrichment (e.g., person attributes, activity detection)",
                            "properties": {
                              "category": {
                                "type": "string",
                                "description": "Object category label (duplicated here for convenience)",
                                "x-parser-schema-id": "<anonymous-schema-67>"
                              },
                              "category_id": {
                                "type": "integer",
                                "description": "Object category ID (duplicated here for convenience)",
                                "x-parser-schema-id": "<anonymous-schema-68>"
                              },
                              "attributes": {
                                "type": "object",
                                "description": "Attribute predictions with confidence per item",
                                "additionalProperties": {
                                  "type": "object",
                                  "required": [
                                    "value",
                                    "confidence"
                                  ],
                                  "properties": {
                                    "value": {
                                      "type": "boolean",
                                      "description": "Binary attribute prediction",
                                      "x-parser-schema-id": "<anonymous-schema-71>"
                                    },
                                    "confidence": {
                                      "type": "number",
                                      "description": "Confidence score for this attribute (0.0 to 1.0)",
                                      "minimum": 0,
                                      "maximum": 1,
                                      "x-parser-schema-id": "<anonymous-schema-72>"
                                    }
                                  },
                                  "x-parser-schema-id": "<anonymous-schema-70>"
                                },
                                "x-parser-schema-id": "<anonymous-schema-69>"
                              }
                            },
                            "additionalProperties": true,
                            "x-parser-schema-id": "<anonymous-schema-66>"
                          }
                        },
                        "additionalProperties": false
                      },
                      "x-parser-schema-id": "<anonymous-schema-63>"
                    },
                    "metadata": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.metadata"
                  },
                  "additionalProperties": false
                },
                "x-parser-schema-id": "<anonymous-schema-57>"
              }
            },
            "additionalProperties": false
          },
          "x-parser-unique-object-id": "snapshot"
        }
      },
      "description": "Device publishes camera snapshots",
      "parameters": {
        "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
      },
      "x-parser-unique-object-id": "deviceSnapshot"
    },
    "metricsCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendMetricsCommand.channel",
    "filterCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendFilterCommand.channel",
    "configCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendConfigCommand.channel",
    "metricsData": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.publishMetricsData.channel",
    "logCommand": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.sendLogCommand.channel",
    "deviceLogs": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations.publishLogEntry.channel",
    "webrtcOffer": {
      "address": "devices/{deviceId}/webrtc/offer",
      "messages": {
        "webrtcSignal": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.components.messages.WebRTCSignal"
      },
      "description": "WebRTC offer from server to device",
      "parameters": {
        "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
      },
      "x-parser-unique-object-id": "webrtcOffer"
    },
    "webrtcAnswer": {
      "address": "devices/{deviceId}/webrtc/answer",
      "messages": {
        "webrtcSignal": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.components.messages.WebRTCSignal"
      },
      "description": "WebRTC answer from device to server",
      "parameters": {
        "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
      },
      "x-parser-unique-object-id": "webrtcAnswer"
    },
    "webrtcIceCandidate": {
      "address": "devices/{deviceId}/webrtc/ice",
      "messages": {
        "webrtcSignal": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.components.messages.WebRTCSignal"
      },
      "description": "ICE candidate exchange",
      "parameters": {
        "deviceId": "$ref:$.channels.deviceRegister.parameters.deviceId"
      },
      "x-parser-unique-object-id": "webrtcIceCandidate"
    }
  },
  "operations": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.operations",
  "components": "$ref:$.channels.deviceSnapshot.messages.snapshot.payload.properties.detections.items.properties.children.items.properties.children.items.components",
  "x-parser-spec-parsed": true,
  "x-parser-api-version": 3,
  "x-parser-circular": true,
  "x-parser-schema-id": "<anonymous-schema-65>",
  "x-parser-spec-stringified": true
};
    const config = {"show":{"sidebar":true},"sidebar":{"showOperations":"byDefault"}};
    const appRoot = document.getElementById('root');
    AsyncApiStandalone.render(
        { schema, config, }, appRoot
    );
  