import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  Globe, 
  Server, 
  Cpu, 
  Layers, 
  Boxes,
  Camera,
  Database,
  Network,
  Code2,
  Brain,
  BookOpen,
  ExternalLink
} from "lucide-react";

export default function Technology() {
  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Technology Overview</h1>
        <p className="text-lg text-muted-foreground">
          Architecture and components powering the Edge AI Platform
        </p>
      </div>

      {/* Web Platform */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <Globe className="h-6 w-6" />
            <div>
              <CardTitle>Web Platform</CardTitle>
              <CardDescription>Browser-based management interface</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Frontend Stack</h4>
            <div className="flex flex-wrap gap-2 mb-3">
              <Badge variant="secondary">React 18</Badge>
              <Badge variant="secondary">TypeScript</Badge>
              <Badge variant="secondary">Vite</Badge>
              <Badge variant="secondary">TailwindCSS</Badge>
              <Badge variant="secondary">shadcn/ui</Badge>
              <Badge variant="secondary">TanStack Query</Badge>
              <Badge variant="secondary">Wouter</Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Modern React application with type-safe APIs, real-time updates via WebSocket, 
              and responsive UI components. Uses TanStack Query for server state management 
              and Wouter for client-side routing.
            </p>
          </div>
          
          <Separator />
          
          <div>
            <h4 className="font-semibold mb-2">Key Features</h4>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
              <li>Real-time device monitoring and alerts</li>
              <li>Search interface with natural language processing</li>
              <li>Live camera snapshots via WebSocket</li>
              <li>Device configuration management</li>
              <li>Detection audit logs and filtering</li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Server Platform */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <Server className="h-6 w-6" />
            <div>
              <CardTitle>Server Platform</CardTitle>
              <CardDescription>Backend services and message broker</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Backend Stack</h4>
            <div className="flex flex-wrap gap-2 mb-3">
              <Badge variant="secondary">Node.js</Badge>
              <Badge variant="secondary">Express.js</Badge>
              <Badge variant="secondary">TypeScript</Badge>
              <Badge variant="secondary">Drizzle ORM</Badge>
              <Badge variant="secondary">PostgreSQL</Badge>
              <Badge variant="secondary">Redis</Badge>
              <Badge variant="secondary">MQTT (Mosquitto)</Badge>
              <Badge variant="secondary">WebSocket (ws)</Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              RESTful API server with real-time messaging, database persistence, and IoT device communication.
              Uses OpenAPI and AsyncAPI specifications for contract-first development.
            </p>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-2">Architecture</h4>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div className="flex items-start gap-2">
                <Database className="h-4 w-4 mt-0.5" />
                <div>
                  <span className="font-medium">PostgreSQL:</span> Device registry, configurations, 
                  search terms, detection audit logs, and version history
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Network className="h-4 w-4 mt-0.5" />
                <div>
                  <span className="font-medium">MQTT:</span> Publish/subscribe messaging for device commands, 
                  status updates, alerts, and snapshots (QoS 0-1)
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Globe className="h-4 w-4 mt-0.5" />
                <div>
                  <span className="font-medium">WebSocket:</span> Real-time browser updates for device events, 
                  metrics, and snapshot streaming
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Code2 className="h-4 w-4 mt-0.5" />
                <div>
                  <span className="font-medium">Validation:</span> Runtime schema validation with AJV for all 
                  MQTT messages and OpenAPI middleware for REST endpoints
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Architecture Diagram */}
      <Card>
        <CardHeader>
          <CardTitle>System Architecture</CardTitle>
          <CardDescription>High-level overview of platform components and communication</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted/30 p-6 rounded-lg">
            <img 
              src="/diagrams/system-architecture.svg" 
              alt="System Architecture Diagram"
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Device Platform */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <Cpu className="h-6 w-6" />
            <div>
              <CardTitle>Device Platform</CardTitle>
              <CardDescription>Edge AI inference on Raspberry Pi, Jetson, and macOS</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Runtime Stack</h4>
            <div className="flex flex-wrap gap-2 mb-3">
              <Badge variant="secondary">Python 3.10+</Badge>
              <Badge variant="secondary">PyTorch</Badge>
              <Badge variant="secondary">ONNX Runtime</Badge>
              <Badge variant="secondary">OpenCV</Badge>
              <Badge variant="secondary">Pydantic V2</Badge>
              <Badge variant="secondary">paho-mqtt</Badge>
              <Badge variant="secondary">Pillow</Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Lightweight vision pipeline optimized for edge devices with hardware acceleration support 
              (CUDA, TensorRT on Jetson, CPU on RPi/macOS).
            </p>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-2">Supported Hardware</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="font-medium">NVIDIA Jetson</p>
                <p className="text-muted-foreground">Orin Nano, Xavier NX</p>
                <p className="text-xs text-muted-foreground">GStreamer + TensorRT</p>
              </div>
              <div>
                <p className="font-medium">Raspberry Pi</p>
                <p className="text-muted-foreground">4B, 5 with IMX500</p>
                <p className="text-xs text-muted-foreground">libcamera + picamera2</p>
              </div>
              <div>
                <p className="font-medium">macOS</p>
                <p className="text-muted-foreground">Development platform</p>
                <p className="text-xs text-muted-foreground">OpenCV camera</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Pipeline Architecture */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <Layers className="h-6 w-6" />
            <div>
              <CardTitle>Pipeline Architecture</CardTitle>
              <CardDescription>Modular task graph for vision processing</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Camera Framework</h4>
            <p className="text-sm text-muted-foreground mb-3">
              Lightweight pipeline framework with task-based architecture. Tasks are connected in a directed 
              graph, data flows through contexts, and execution is managed by a runner with cooperative scheduling.
            </p>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs space-y-1">
              <div>Camera → YOLO Detection → Attribute Classification</div>
              <div className="ml-4">→ Color Extraction → Clusterer → Tracker</div>
              <div className="ml-8">→ Filter → Alert Sink</div>
              <div className="ml-8">→ MQTT Sink (snapshot + metadata)</div>
            </div>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-2">Key Concepts</h4>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div>
                <span className="font-medium">Tasks:</span> Modular processing units (e.g., camera capture, 
                object detection, tracking). Each task declares inputs/outputs via contracts.
              </div>
              <div>
                <span className="font-medium">Context:</span> Data container passing between tasks. 
                Holds image frames, detections, metadata, and exception state.
              </div>
              <div>
                <span className="font-medium">Runner:</span> Executes task graph with time budgets, 
                fork/merge support for parallel branches, and exception propagation.
              </div>
              <div>
                <span className="font-medium">Sinks:</span> Output adapters (MQTT publish, file storage, 
                display) that consume pipeline results.
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Pipeline Architecture Diagram */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <Layers className="h-6 w-6" />
            <div>
              <CardTitle>Pipeline Flow</CardTitle>
              <CardDescription>15 tasks connected via 9 buffers</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="bg-muted/30 p-6 rounded-lg">
            <img 
              src="/diagrams/pipeline-topology.svg" 
              alt="Pipeline Topology Diagram"
              className="w-full"
            />
          </div>
          <div className="mt-4 text-xs text-muted-foreground">
            <p className="font-medium mb-2">Pipeline Stages:</p>
            <ul className="space-y-1 list-disc list-inside">
              <li><span className="font-mono bg-blue-100 px-1 rounded">Camera Tasks</span> - Image capture and initial processing</li>
              <li><span className="font-mono bg-purple-100 px-1 rounded">Vision Tasks</span> - Object detection, attributes, VLM verification</li>
              <li><span className="font-mono bg-green-100 px-1 rounded">MQTT Tasks</span> - Device communication and publishing</li>
              <li><span className="font-mono bg-yellow-100 px-1 rounded">Buffers</span> - Thread-safe queues between tasks</li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Tasks and Models */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <Boxes className="h-6 w-6" />
            <div>
              <CardTitle>Tasks & Models</CardTitle>
              <CardDescription>Computer vision and ML inference components</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <Camera className="h-4 w-4" />
              Object Detection
            </h4>
            <div className="bg-muted/50 p-3 rounded space-y-2 text-sm">
              <div>
                <span className="font-medium">YOLO (YOLOv8):</span>
                <p className="text-muted-foreground">
                  Real-time object detection with 80 COCO categories (person, car, bicycle, etc.). 
                  Runs on PyTorch or TensorRT for GPU acceleration. Outputs bounding boxes with confidence scores.
                </p>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                <Badge variant="outline" className="text-xs">Confidence threshold: 0.25</Badge>
                <Badge variant="outline" className="text-xs">IoU threshold: 0.45</Badge>
                <Badge variant="outline" className="text-xs">80 categories</Badge>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Attribute Classification
            </h4>
            <div className="bg-muted/50 p-3 rounded space-y-2 text-sm">
              <div>
                <span className="font-medium">PA100K Model:</span>
                <p className="text-muted-foreground">
                  Pedestrian attribute recognition (26 attributes): clothing colors, accessories (hat, glasses), 
                  garment types (long coat, dress, shorts). ONNX runtime for cross-platform inference.
                </p>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                <Badge variant="outline" className="text-xs">Upper body colors</Badge>
                <Badge variant="outline" className="text-xs">Lower body colors</Badge>
                <Badge variant="outline" className="text-xs">Accessories</Badge>
                <Badge variant="outline" className="text-xs">Garment types</Badge>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <Code2 className="h-4 w-4" />
              Color Extraction
            </h4>
            <div className="bg-muted/50 p-3 rounded space-y-2 text-sm">
              <div>
                <span className="font-medium">Region-based Analysis:</span>
                <p className="text-muted-foreground">
                  Extracts dominant colors from specific regions of detections (hat, upper body, lower body, boots). 
                  Uses HSV color space with named color matching and ellipse-based region cropping.
                </p>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2">Clustering & Tracking</h4>
            <div className="bg-muted/50 p-3 rounded space-y-2 text-sm">
              <div>
                <span className="font-medium">Clusterer:</span> Groups similar detections using proximity, 
                size, category, and attribute similarity with configurable merge thresholds.
              </div>
              <div>
                <span className="font-medium">Tracker:</span> Maintains object identity across frames using 
                IoU matching, attribute similarity, and confirmation windows. Includes cooldown and TTL 
                for lifecycle management.
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-2">Search & Filtering</h4>
            <p className="text-sm text-muted-foreground">
              Natural language search queries are processed by Groq (Llama 3.1) to generate category masks 
              (80 booleans), attribute masks (26 booleans), and color requirements. Devices filter detections 
              in real-time against these masks before sending alerts.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* API Documentation */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <BookOpen className="h-6 w-6" />
            <div>
              <CardTitle>API Documentation</CardTitle>
              <CardDescription>Interactive API references and specifications</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Available Documentation</h4>
            <div className="space-y-3">
              <a 
                href="/docs/openapi/index.html"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-3 rounded-lg border hover:bg-muted/50 transition-colors"
              >
                <div className="flex-1">
                  <div className="font-medium text-sm">REST API (OpenAPI 3.1)</div>
                  <div className="text-xs text-muted-foreground">
                    Interactive documentation for HTTP endpoints - device management, search, configuration
                  </div>
                </div>
                <ExternalLink className="h-4 w-4 text-muted-foreground" />
              </a>
              
              <a 
                href="/docs/asyncapi-mqtt/index.html"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-3 rounded-lg border hover:bg-muted/50 transition-colors"
              >
                <div className="flex-1">
                  <div className="font-medium text-sm">MQTT API (AsyncAPI 3.0)</div>
                  <div className="text-xs text-muted-foreground">
                    Device-to-server messaging - register, heartbeat, status, alerts, snapshots, metrics, commands
                  </div>
                </div>
                <ExternalLink className="h-4 w-4 text-muted-foreground" />
              </a>
              
              <a 
                href="/docs/asyncapi-websocket/index.html"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-3 rounded-lg border hover:bg-muted/50 transition-colors"
              >
                <div className="flex-1">
                  <div className="font-medium text-sm">WebSocket API (AsyncAPI 3.0)</div>
                  <div className="text-xs text-muted-foreground">
                    Browser real-time updates - connection events, snapshots, metrics, logs, alerts
                  </div>
                </div>
                <ExternalLink className="h-4 w-4 text-muted-foreground" />
              </a>
            </div>
          </div>
          
          <Separator />
          
          <div>
            <h4 className="font-semibold mb-2 text-sm">Specification Files</h4>
            <div className="space-y-2 text-xs text-muted-foreground">
              <div>
                <span className="font-medium font-mono">shared/specs/openapi.yaml</span> - REST API contract
              </div>
              <div>
                <span className="font-medium font-mono">shared/specs/asyncapi-mqtt.yaml</span> - MQTT topics and schemas
              </div>
              <div>
                <span className="font-medium font-mono">shared/specs/asyncapi-websocket.yaml</span> - WebSocket message envelope
              </div>
              <div>
                <span className="font-medium font-mono">shared/schemas/</span> - Shared JSON schemas (register, snapshot, alerts, metrics, etc.)
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Contract-First Development */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <Code2 className="h-6 w-6" />
            <div>
              <CardTitle>Contract-First Development</CardTitle>
              <CardDescription>Schema-driven code generation</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">API Specifications</h4>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div>
                <span className="font-medium">OpenAPI 3.1:</span> REST API contract with automated validation 
                middleware. TypeScript types generated via openapi-typescript.
              </div>
              <div>
                <span className="font-medium">AsyncAPI 3.0:</span> MQTT and WebSocket message contracts. 
                Separate specs for device communication and browser updates.
              </div>
              <div>
                <span className="font-medium">JSON Schema:</span> Shared message schemas for register, snapshot, 
                alerts, metrics, etc. Python models generated via datamodel-codegen.
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-2">Code Generation Flow</h4>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs space-y-1">
              <div>1. Edit JSON schema: shared/schemas/*.json</div>
              <div>2. Update AsyncAPI/OpenAPI specs</div>
              <div>3. Run: just generate</div>
              <div>4. TypeScript types → web-platform/shared/types/api.ts</div>
              <div>5. Python models → device-sdk/edge_llm_client/models/</div>
              <div>6. Runtime validation with AJV (server) and Pydantic (device)</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
