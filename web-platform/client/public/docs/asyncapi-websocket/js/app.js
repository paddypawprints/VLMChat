
    const schema = {
  "asyncapi": "3.0.0",
  "info": {
    "title": "Edge AI Platform WebSocket API",
    "version": "1.0.0",
    "description": "WebSocket messaging protocol for browser clients connecting to the Edge AI Platform.\nEnables real-time device monitoring, snapshot viewing, and metrics streaming.\n\nThis API is separate from the MQTT API (asyncapi-mqtt.yaml) which handles device-server communication.\nThe server acts as a bridge, translating WebSocket client messages to MQTT commands and vice versa.\n",
    "contact": {
      "name": "VLMChat Platform"
    },
    "license": {
      "name": "MIT"
    }
  },
  "servers": {
    "development": {
      "host": "localhost:3000/ws",
      "protocol": "ws",
      "description": "Local development WebSocket server",
      "tags": [
        {
          "name": "env:development"
        }
      ]
    },
    "production": {
      "host": "platform.example.com/ws",
      "protocol": "wss",
      "description": "Production WebSocket server (TLS)",
      "tags": [
        {
          "name": "env:production"
        }
      ]
    }
  },
  "defaultContentType": "application/json",
  "channels": {
    "websocket": {
      "address": "/",
      "messages": {
        "envelope": {
          "name": "WebSocketEnvelope",
          "title": "WebSocket Message Envelope",
          "summary": "All WebSocket messages use this envelope wrapper",
          "contentType": "application/json",
          "payload": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$comment": "⚠️  SECURITY CRITICAL - DO NOT MODIFY WITHOUT REVIEW. This schema validates WebSocket messages and protects against malformed or malicious data. Changes must be coordinated across all validation layers (server WebSocket validator, browser validator, API specs).",
            "$id": "websocket-envelope-v1.0.0.json",
            "title": "WebSocket Message Envelope",
            "description": "Envelope wrapper for WebSocket messages that contains a type discriminator and the actual message payload. The message field contains one of the protocol-specific message schemas.",
            "type": "object",
            "required": [
              "type",
              "message"
            ],
            "properties": {
              "type": {
                "type": "string",
                "description": "Message type discriminator that determines which schema validates the message field",
                "enum": [
                  "metrics_start",
                  "metrics_stop",
                  "logs_start",
                  "logs_stop",
                  "connected",
                  "snapshot",
                  "metrics",
                  "logs",
                  "alert"
                ],
                "x-parser-schema-id": "<anonymous-schema-1>"
              },
              "message": {
                "type": "object",
                "description": "The actual message payload, validated according to the type field",
                "x-parser-schema-id": "<anonymous-schema-2>"
              }
            },
            "additionalProperties": false,
            "oneOf": [
              {
                "properties": {
                  "type": {
                    "const": "metrics_start"
                  },
                  "message": {
                    "type": "object",
                    "additionalProperties": false
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-3>"
              },
              {
                "properties": {
                  "type": {
                    "const": "metrics_stop"
                  },
                  "message": {
                    "type": "object",
                    "additionalProperties": false
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-4>"
              },
              {
                "properties": {
                  "type": {
                    "const": "connected"
                  },
                  "message": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$comment": "⚠️  SECURITY CRITICAL - DO NOT MODIFY WITHOUT REVIEW. This schema validates WebSocket messages and protects against malformed or malicious data. Changes must be coordinated across all validation layers (server WebSocket validator, browser validator, API specs).",
                    "$id": "connection-ack-v1.0.0.json",
                    "title": "Connection Acknowledgment",
                    "description": "Server acknowledgment of successful WebSocket connection",
                    "type": "object",
                    "required": [
                      "deviceId",
                      "timestamp"
                    ],
                    "properties": {
                      "deviceId": {
                        "type": "string",
                        "description": "Device ID for this connection",
                        "minLength": 1
                      },
                      "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Server timestamp when connection established"
                      }
                    },
                    "additionalProperties": false
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-5>"
              },
              {
                "properties": {
                  "type": {
                    "const": "snapshot"
                  },
                  "message": {
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
                        "minLength": 1
                      },
                      "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "ISO 8601 UTC timestamp when the snapshot was captured"
                      },
                      "image": {
                        "type": "string",
                        "description": "Base64-encoded image data",
                        "minLength": 1
                      },
                      "format": {
                        "type": "string",
                        "enum": [
                          "jpeg",
                          "png"
                        ],
                        "description": "Image format",
                        "default": "jpeg"
                      },
                      "width": {
                        "type": "integer",
                        "description": "Image width in pixels",
                        "minimum": 1
                      },
                      "height": {
                        "type": "integer",
                        "description": "Image height in pixels",
                        "minimum": 1
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
                                "type": "number"
                              },
                              "minItems": 4,
                              "maxItems": 4
                            },
                            "confidence": {
                              "type": "number",
                              "description": "Detection confidence score (0.0 to 1.0)",
                              "minimum": 0,
                              "maximum": 1
                            },
                            "category": {
                              "type": "string",
                              "description": "Object category label (e.g., 'person', 'car', 'dog')",
                              "minLength": 1
                            },
                            "category_id": {
                              "type": "integer",
                              "description": "COCO category ID (-1 for unknown)"
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
                                "required": "$ref:$.channels.websocket.messages.envelope.payload.oneOf[3].properties.message.properties.detections.items.required",
                                "properties": {
                                  "bbox": "$ref:$.channels.websocket.messages.envelope.payload.oneOf[3].properties.message.properties.detections.items.properties.bbox",
                                  "confidence": "$ref:$.channels.websocket.messages.envelope.payload.oneOf[3].properties.message.properties.detections.items.properties.confidence",
                                  "category": "$ref:$.channels.websocket.messages.envelope.payload.oneOf[3].properties.message.properties.detections.items.properties.category",
                                  "category_id": "$ref:$.channels.websocket.messages.envelope.payload.oneOf[3].properties.message.properties.detections.items.properties.category_id",
                                  "children": {
                                    "type": "array",
                                    "description": "Child detections (for clustered objects)",
                                    "items": {
                                      "asyncapi": "3.0.0",
                                      "info": "$ref:$.info",
                                      "servers": "$ref:$.servers",
                                      "defaultContentType": "application/json",
                                      "channels": "$ref:$.channels",
                                      "components": {
                                        "messages": {
                                          "WebSocketEnvelope": "$ref:$.channels.websocket.messages.envelope"
                                        }
                                      },
                                      "x-parser-spec-parsed": true,
                                      "x-parser-api-version": 3,
                                      "x-parser-circular": true
                                    }
                                  },
                                  "metadata": {
                                    "type": "object",
                                    "description": "Extensible metadata for task enrichment (e.g., person attributes, activity detection)",
                                    "properties": {
                                      "category": {
                                        "type": "string",
                                        "description": "Object category label (duplicated here for convenience)"
                                      },
                                      "category_id": {
                                        "type": "integer",
                                        "description": "Object category ID (duplicated here for convenience)"
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
                                              "description": "Binary attribute prediction"
                                            },
                                            "confidence": {
                                              "type": "number",
                                              "description": "Confidence score for this attribute (0.0 to 1.0)",
                                              "minimum": 0,
                                              "maximum": 1
                                            }
                                          }
                                        }
                                      }
                                    },
                                    "additionalProperties": true
                                  }
                                },
                                "additionalProperties": false
                              }
                            },
                            "metadata": "$ref:$.channels.websocket.messages.envelope.payload.oneOf[3].properties.message.properties.detections.items.properties.children.items.properties.metadata"
                          },
                          "additionalProperties": false
                        }
                      }
                    },
                    "additionalProperties": false
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-6>"
              },
              {
                "properties": {
                  "type": {
                    "const": "metrics"
                  },
                  "message": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$comment": "⚠️  SECURITY CRITICAL - DO NOT MODIFY WITHOUT REVIEW. This schema validates incoming MQTT messages and protects against malformed or malicious data. Changes must be coordinated across all validation layers (server MQTT validator, device validator, API specs).",
                    "$id": "metrics-data-v1.0.0.json",
                    "title": "Metrics Data",
                    "description": "Instrument values from a metrics session",
                    "type": "object",
                    "required": [
                      "session",
                      "timestamp",
                      "instruments"
                    ],
                    "properties": {
                      "session": {
                        "type": "string",
                        "description": "Session name"
                      },
                      "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Data snapshot timestamp"
                      },
                      "instruments": {
                        "type": "array",
                        "items": {
                          "type": "object",
                          "required": [
                            "name"
                          ],
                          "properties": {
                            "name": {
                              "type": "string",
                              "description": "Instrument name"
                            },
                            "type": {
                              "type": "string",
                              "description": "Instrument type"
                            },
                            "value": {
                              "description": "Current instrument value (varies by type)"
                            }
                          }
                        }
                      }
                    }
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-7>"
              },
              {
                "properties": {
                  "type": {
                    "const": "logs_start"
                  },
                  "message": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$id": "websocket-log-start-v1.0.0.json",
                    "title": "WebSocket Log Start Command",
                    "description": "Browser request to start log streaming from device",
                    "type": "object",
                    "properties": {
                      "level": {
                        "type": "string",
                        "description": "Minimum log level to stream",
                        "enum": [
                          "DEBUG",
                          "INFO",
                          "WARNING",
                          "ERROR",
                          "CRITICAL"
                        ]
                      }
                    },
                    "additionalProperties": false
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-8>"
              },
              {
                "properties": {
                  "type": {
                    "const": "logs_stop"
                  },
                  "message": {
                    "type": "object",
                    "additionalProperties": false
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-9>"
              },
              {
                "properties": {
                  "type": {
                    "const": "logs"
                  },
                  "message": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$id": "log-entry-v1.0.0.json",
                    "title": "Log Entry",
                    "description": "A single log entry from a device",
                    "type": "object",
                    "properties": {
                      "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "ISO 8601 timestamp of the log entry"
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
                        "description": "Log level"
                      },
                      "logger": {
                        "type": "string",
                        "description": "Logger name (e.g., module or component name)"
                      },
                      "message": {
                        "type": "string",
                        "description": "Log message"
                      },
                      "module": {
                        "type": "string",
                        "description": "Module or file where log originated"
                      },
                      "line": {
                        "type": "integer",
                        "description": "Line number in source file"
                      },
                      "thread": {
                        "type": "string",
                        "description": "Thread name or ID"
                      }
                    },
                    "required": [
                      "timestamp",
                      "level",
                      "logger",
                      "message"
                    ],
                    "additionalProperties": false
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-10>"
              },
              {
                "properties": {
                  "type": {
                    "const": "alert"
                  },
                  "message": {
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
                        "description": "Alert type"
                      },
                      "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "When the alert occurred"
                      },
                      "watchlist_item_id": {
                        "type": "string",
                        "description": "ID of matched watchlist item (for detection alerts)"
                      },
                      "description": {
                        "type": "string",
                        "description": "Alert description"
                      },
                      "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Detection confidence score"
                      },
                      "image": {
                        "type": "string",
                        "description": "Base64 encoded image or reference"
                      },
                      "image_url": {
                        "type": "string",
                        "format": "uri",
                        "description": "URL to image in storage"
                      },
                      "metadata": {
                        "type": "object",
                        "description": "Additional alert metadata",
                        "properties": {
                          "bounding_box": {
                            "type": "object",
                            "properties": {
                              "x": {
                                "type": "number"
                              },
                              "y": {
                                "type": "number"
                              },
                              "width": {
                                "type": "number"
                              },
                              "height": {
                                "type": "number"
                              }
                            }
                          },
                          "inference_time_ms": {
                            "type": "number",
                            "description": "Time taken for inference"
                          }
                        }
                      }
                    }
                  }
                },
                "x-parser-schema-id": "<anonymous-schema-11>"
              }
            ]
          },
          "x-parser-unique-object-id": "envelope"
        }
      },
      "description": "Bidirectional WebSocket channel using envelope pattern.\nAll messages wrapped in {type: string, message: object} envelope.\n\nClient→Server types: metrics_start, metrics_stop, logs_start, logs_stop\nServer→Client types: connected, snapshot, metrics, logs, alert\n",
      "x-parser-unique-object-id": "websocket"
    }
  },
  "components": "$ref:$.channels.websocket.messages.envelope.payload.oneOf[3].properties.message.properties.detections.items.properties.children.items.properties.children.items.components",
  "x-parser-spec-parsed": true,
  "x-parser-api-version": 3,
  "x-parser-circular": true,
  "x-parser-spec-stringified": true
};
    const config = {"show":{"sidebar":true},"sidebar":{"showOperations":"byDefault"}};
    const appRoot = document.getElementById('root');
    AsyncApiStandalone.render(
        { schema, config, }, appRoot
    );
  