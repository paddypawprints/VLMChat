import { useParams } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useRef, useCallback } from "react";
import { apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { SnapshotCanvas } from "@/components/SnapshotCanvas";
import { VlmStatusBadge } from "@/components/VlmStatusBadge";
import { 
  ArrowLeft, 
  RefreshCw, 
  Camera,
  Cpu,
  HardDrive,
  Thermometer,
  Zap,
  Activity,
  Clock,
  Video,
  VideoOff,
  AlertCircle,
  Info
} from "lucide-react";
import { Link } from "wouter";
import { useWebSocket } from "@/hooks/use-websocket";
import { useWebRTC } from "@/hooks/use-webrtc";
import { FpsGauge } from "@/components/metrics/FpsGauge";
import { DurationChart } from "@/components/metrics/DurationChart";
import { MemoryLeakChart } from "@/components/metrics/MemoryLeakChart";
import { LogViewer } from "@/components/logs/LogViewer";

interface DeviceSpecs {
  cpu?: string;
  memory?: string;
  temperature?: number;
  usage?: number;
  [key: string]: any;
}

interface Device {
  deviceId: string;
  name: string;
  status: string;
  userId?: string | null;
  lastSeen?: string | null;
  createdAt?: string;
}

interface MetricsData {
  session: string;
  timestamp: string;
  instruments: Array<{
    name: string;
    type: string;
    value: any;
  }>;
}

interface DataPoint {
  timestamp: string;
  value: number;
}

interface MemoryDataPoint {
  timestamp: string;
  alive: number;
  cleaned: number;
  leaked: number;
}

interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  module?: string;
  line?: number;
  thread?: string;
}

export default function DeviceDetails() {
  const { id } = useParams<{ id: string }>();
  const [snapshot, setSnapshot] = useState<string | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [snapshotTimestamp, setSnapshotTimestamp] = useState<string | null>(null);
  const [detections, setDetections] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [durationHistory, setDurationHistory] = useState<DataPoint[]>([]);
  const [fpsHistory, setFpsHistory] = useState<DataPoint[]>([]);
  const [memoryHistory, setMemoryHistory] = useState<MemoryDataPoint[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logLevel, setLogLevel] = useState<string>("WARNING");
  const [alerts, setAlerts] = useState<any[]>([]);
  const [selectedAlertMetadata, setSelectedAlertMetadata] = useState<any | null>(null);

  // Stable callback for snapshot handling
  const handleSnapshot = useCallback((data: any) => {
    console.log('[Snapshot] Received via WebSocket');
    console.log('[Snapshot] Setting state - image length:', data.image?.length, 'timestamp:', data.timestamp, 'detections:', data.detections?.length);
    setSnapshot(data.image);
    setSnapshotTimestamp(data.timestamp);
    setDetections(data.detections || []);
    setSnapshotLoading(false);
    console.log('[Snapshot] State updated');
  }, []);

  // Stable callback for metrics handling
  const handleMetrics = useCallback((data: MetricsData) => {
    console.log('[Metrics] Received via WebSocket:', data);
    setMetrics(data);
    
    // Update time series history
    const timestamp = data.timestamp;
    
    // Find FPS instrument
    const fpsInstrument = data.instruments.find(i => i.type === 'rate');
    if (fpsInstrument && typeof fpsInstrument.value === 'number') {
      setFpsHistory(prev => {
        const newHistory = [...prev, { timestamp, value: fpsInstrument.value }];
        // Keep last 50 points
        return newHistory.slice(-50);
      });
    }
    
    // Find duration instrument
    const durationInstrument = data.instruments.find(i => i.type === 'avg_duration');
    if (durationInstrument && typeof durationInstrument.value === 'number') {
      setDurationHistory(prev => {
        const newHistory = [...prev, { timestamp, value: durationInstrument.value }];
        // Keep last 50 points
        return newHistory.slice(-50);
      });
    }
    
    // Find memory instruments (alive, cleaned, leaked)
    const aliveInstrument = data.instruments.find(i => i.name.endsWith('.alive'));
    const cleanedInstrument = data.instruments.find(i => i.name.endsWith('.cleaned'));
    const leakedInstrument = data.instruments.find(i => i.name.endsWith('.leaked'));
    
    if (aliveInstrument && cleanedInstrument) {
      setMemoryHistory(prev => {
        const newHistory = [...prev, {
          timestamp,
          alive: typeof aliveInstrument.value === 'number' ? aliveInstrument.value : 0,
          cleaned: typeof cleanedInstrument.value === 'number' ? cleanedInstrument.value : 0,
          leaked: leakedInstrument && typeof leakedInstrument.value === 'number' ? leakedInstrument.value : 0,
        }];
        // Keep last 50 points
        return newHistory.slice(-50);
      });
    }
  }, []);

  // Stable callback for log handling
  const handleLog = useCallback((data: LogEntry) => {
    setLogs(prev => {
      const newLogs = [...prev, data];
      // Keep last 200 log entries
      return newLogs.slice(-200);
    });
  }, []);

  // Stable callback for alert handling
  const handleAlert = useCallback((data: any) => {
    console.log('[Alert] Received via WebSocket:', data);
    setAlerts(prev => {
      const newAlerts = [data, ...prev];
      // Keep last 50 alerts
      return newAlerts.slice(0, 50);
    });
  }, []);

  // Connect to WebSocket for real-time snapshot delivery and metrics
  const { isConnected: wsConnected, send } = useWebSocket({
    deviceId: id || '',
    enabled: !!id,
    onSnapshot: handleSnapshot,
    onMetrics: handleMetrics,
    onLogs: handleLog,
    onAlert: handleAlert
  });

  // Handle log level changes
  const handleLogLevelChange = useCallback((level: string) => {
    setLogLevel(level);
    setLogs([]); // Clear existing logs when changing level
    if (send) {
      send({ type: 'logs_start', message: { level } });
    }
  }, [send]);

  // Start log streaming on mount
  useEffect(() => {
    if (send && id) {
      send({ type: 'logs_start', message: { level: logLevel } });
    }
    // Stop log streaming on unmount
    return () => {
      if (send) {
        send({ type: 'logs_stop', message: {} });
      }
    };
  }, [send, id, logLevel]);

  // Fetch device details
  const { data: device, isLoading: deviceLoading } = useQuery<Device>({
    queryKey: [`/api/devices/${id}`],
    enabled: !!id,
  });

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'raspberry-pi': return '🔴';
      case 'jetson': return '🟢';
      case 'coral': return '🟡';
      case 'ncs': return '🔵';
      case 'mac': return '💻';
      default: return '⚪';
    }
  };

  const isDeviceOnline = (device: Device | undefined) => {
    if (!device) return false;
    // Use status field which is enriched with Redis real-time data
    return device.status === 'connected';
  };

  const requestSnapshot = async () => {
    if (!id) return;
    setSnapshotLoading(true);
    try {
      // Request snapshot from backend (will send MQTT command to device)
      await apiRequest('POST', `/api/devices/${id}/snapshot`);
      console.log('[Snapshot] Request sent, waiting for WebSocket delivery...');
      // Snapshot will be delivered via WebSocket (handled by useWebSocket hook)
    } catch (error) {
      console.error('Failed to request snapshot:', error);
      setSnapshotLoading(false);
    }
  };

  if (deviceLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center gap-4 mb-6">
          <Link href="/devices">
            <Button variant="ghost" size="sm" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          </Link>
        </div>
        <div className="grid gap-6">
          <Card>
            <CardContent className="p-6">
              <p>Loading device details...</p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (!device) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center gap-4 mb-6">
          <Link href="/devices">
            <Button variant="ghost" size="sm" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          </Link>
        </div>
        <Card>
          <CardContent className="p-6">
            <p className="text-destructive">Device not found</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Link href="/devices">
            <Button variant="ghost" size="sm" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          </Link>
          <div className="flex items-center gap-3">
            <span className="text-3xl">💻</span>
            <div>
              <h1 className="text-3xl font-bold">{device.name}</h1>
              <p className="text-muted-foreground">{device.deviceId}</p>
            </div>
          </div>
        </div>
        <Badge variant={isDeviceOnline(device) ? 'default' : 'secondary'}>
          {isDeviceOnline(device) ? 'Online' : 'Offline'}
        </Badge>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Device Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Device Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Device ID</p>
                <p className="font-medium">{device.deviceId}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Status</p>
                <p className="font-medium capitalize">{device.status}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Last Seen</p>
                <p className="font-medium text-sm">
                  {device.lastSeen 
                    ? new Date(device.lastSeen).toLocaleString()
                    : 'Never'}
                </p>
              </div>
            </div>

            {device.specs && (
              <>
                <Separator />
                <div className="space-y-3">
                  <h4 className="font-semibold text-sm">Hardware Specs</h4>
                  {device.specs.cpu && (
                    <div className="flex items-center gap-2">
                      <Cpu className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">CPU:</span>
                      <span className="text-sm">{device.specs.cpu}</span>
                    </div>
                  )}
                  {device.specs.memory && (
                    <div className="flex items-center gap-2">
                      <HardDrive className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">Memory:</span>
                      <span className="text-sm">{device.specs.memory}</span>
                    </div>
                  )}
                  {device.specs.temperature !== undefined && (
                    <div className="flex items-center gap-2">
                      <Thermometer className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">Temperature:</span>
                      <span className="text-sm">{device.specs.temperature}°C</span>
                    </div>
                  )}
                  {device.specs.usage !== undefined && (
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">CPU Usage:</span>
                      <span className="text-sm">{device.specs.usage}%</span>
                    </div>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Snapshot */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Camera Snapshot
                {wsConnected && (
                  <Badge variant="outline" className="ml-2 text-xs">
                    Live
                  </Badge>
                )}
              </CardTitle>
              <Button
                size="sm"
                onClick={requestSnapshot}
                disabled={snapshotLoading || !isDeviceOnline(device)}
                className="gap-2"
              >
                <RefreshCw className={`h-4 w-4 ${snapshotLoading ? 'animate-spin' : ''}`} />
                {snapshotLoading ? 'Loading...' : 'Refresh'}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
              {snapshot ? (
                <SnapshotCanvas 
                  imageData={snapshot}
                  detections={detections}
                />
              ) : (
                <div className="text-center text-muted-foreground">
                  <Camera className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">
                    {isDeviceOnline(device) 
                      ? 'Click Refresh to capture snapshot'
                      : 'Device offline'}
                  </p>
                </div>
              )}
            </div>
            {snapshotTimestamp && (
              <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Last snapshot: {new Date(snapshotTimestamp).toLocaleString()}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Alerts */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              Recent Alerts
              {alerts.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {alerts.length}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {alerts.slice(0, 3).map((alert, idx) => (
                  <div
                    key={`${alert.timestamp}-${idx}`}
                    className="border rounded-lg overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
                  >
                    <div className="relative bg-muted">
                      {alert.image && (
                        <img
                          src={`data:image/jpeg;base64,${alert.image}`}
                          alt={alert.description}
                          className="w-full h-[200px] object-contain"
                        />
                      )}
                      <Badge
                        variant="default"
                        className="absolute top-2 left-2 bg-black/70 text-white text-lg font-bold px-3 py-1"
                      >
                        {Math.round(alert.confidence * 100)}%
                      </Badge>
                      {/* VLM Status Badge */}
                      <div className="absolute top-2 right-2">
                        <VlmStatusBadge
                          vlmRequired={alert.vlm_required}
                          vlmStatus={alert.vlm_status}
                          vlmResponse={alert.vlm_response}
                        />
                      </div>
                    </div>
                    <div className="p-4 space-y-2">
                      {/* Search String */}
                      {alert.search_string && (
                        <p className="text-xs text-blue-600 dark:text-blue-400 font-medium">
                          🔍 "{alert.search_string}"
                        </p>
                      )}
                      <p className="text-sm font-medium line-clamp-2">{alert.description}</p>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1 text-sm text-muted-foreground">
                          <Clock className="h-4 w-4" />
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </div>
                        {/* Metadata Info Button */}
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0"
                              onClick={() => setSelectedAlertMetadata(alert.metadata)}
                            >
                              <Info className="h-4 w-4" />
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                            <DialogHeader>
                              <DialogTitle>Alert Metadata</DialogTitle>
                              <DialogDescription>
                                Detection details and tracking information
                              </DialogDescription>
                            </DialogHeader>
                            <div className="space-y-4">
                              {/* Bounding Box */}
                              {alert.metadata?.bounding_box && (
                                <div>
                                  <h3 className="font-semibold mb-2">Bounding Box</h3>
                                  <div className="bg-muted p-3 rounded font-mono text-sm">
                                    <div>x: {alert.metadata.bounding_box.x?.toFixed(2)}</div>
                                    <div>y: {alert.metadata.bounding_box.y?.toFixed(2)}</div>
                                    <div>width: {alert.metadata.bounding_box.width?.toFixed(2)}</div>
                                    <div>height: {alert.metadata.bounding_box.height?.toFixed(2)}</div>
                                  </div>
                                </div>
                              )}
                              
                              {/* Track Info */}
                              {alert.metadata?.track_id && (
                                <div>
                                  <h3 className="font-semibold mb-2">Tracking</h3>
                                  <div className="bg-muted p-3 rounded font-mono text-sm space-y-1">
                                    <div>Track ID: {alert.metadata.track_id}</div>
                                    {alert.metadata.confirmation_count && (
                                      <div>Confirmations: {alert.metadata.confirmation_count}</div>
                                    )}
                                    {alert.metadata.first_seen && (
                                      <div>First Seen: {new Date(alert.metadata.first_seen * 1000).toLocaleString()}</div>
                                    )}
                                  </div>
                                </div>
                              )}
                              
                              {/* Attributes */}
                              {alert.metadata?.attributes && Object.keys(alert.metadata.attributes).length > 0 && (
                                <div>
                                  <h3 className="font-semibold mb-2">Attributes</h3>
                                  <div className="bg-muted p-3 rounded text-sm max-h-60 overflow-y-auto">
                                    {Object.entries(alert.metadata.attributes).map(([key, val]: [string, any]) => (
                                      <div key={key} className="flex justify-between py-1 border-b border-border/50 last:border-0">
                                        <span className="font-medium">{key}:</span>
                                        <span className="text-muted-foreground">
                                          {val?.value?.toString()} ({val?.confidence ? (val.confidence * 100).toFixed(1) : '0'}%)
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              
                              {/* Colors */}
                              {alert.metadata?.colors && Object.keys(alert.metadata.colors).length > 0 && (
                                <div>
                                  <h3 className="font-semibold mb-2">Colors</h3>
                                  <div className="bg-muted p-3 rounded text-sm">
                                    {Object.entries(alert.metadata.colors).map(([region, color]: [string, any]) => (
                                      <div key={region} className="flex items-center gap-2 py-1">
                                        <span className="font-medium">{region}:</span>
                                        <div 
                                          className="w-6 h-6 border border-border rounded"
                                          style={{ backgroundColor: `rgb(${color.join(',')})` }}
                                        />
                                        <span className="text-muted-foreground">{color.join(', ')}</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              
                              {/* Raw Metadata JSON */}
                              <div>
                                <h3 className="font-semibold mb-2">Full Metadata (JSON)</h3>
                                <pre className="bg-muted p-3 rounded text-xs overflow-x-auto">
                                  {JSON.stringify(alert.metadata, null, 2)}
                                </pre>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                      </div>
                      {/* Show VLM response for invalid responses */}
                      {alert.vlm_status === 'invalid_response' && alert.vlm_response && (
                        <p className="text-xs text-red-600 dark:text-red-400">
                          ⚠ Invalid VLM response: {alert.vlm_response}
                        </p>
                      )}
                      {/* Show inference time if available */}
                      {alert.vlm_inference_time && (
                        <p className="text-xs text-muted-foreground">
                          VLM: {alert.vlm_inference_time.toFixed(2)}s
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                <AlertCircle className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No alerts yet</p>
                <p className="text-sm mt-1">
                  Alerts will appear here when detections are confirmed
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Metrics */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Performance Metrics
              {wsConnected && <span className="ml-2 h-2 w-2 rounded-full bg-green-500 animate-pulse" title="Live updates"></span>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {metrics ? (
              <div className="space-y-6">
                {/* FPS Gauge and Duration Chart side by side */}
                <div className="grid gap-4 md:grid-cols-2">
                  {(() => {
                    const fpsInstrument = metrics.instruments.find(i => i.type === 'rate');
                    return fpsInstrument ? (
                      <FpsGauge value={fpsInstrument.value} max={60} />
                    ) : null;
                  })()}
                  
                  {durationHistory.length > 0 && (
                    <DurationChart data={durationHistory} title="Pipeline Duration Over Time" />
                  )}
                </div>
                
                {/* Memory Leak Chart */}
                {memoryHistory.length > 0 && (
                  <div className="grid gap-4">
                    <MemoryLeakChart data={memoryHistory} title="Memory Tracking (Long-Lived Objects)" />
                  </div>
                )}

                <div className="flex items-center gap-2 text-sm text-muted-foreground pt-2">
                  <Clock className="h-4 w-4" />
                  <span>Last Updated: {new Date(metrics.timestamp).toLocaleString()}</span>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                <Activity className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No metrics available</p>
                <p className="text-sm mt-1">
                  {isDeviceOnline(device) 
                    ? 'Waiting for device to send metrics...'
                    : 'Device is offline'}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Log Viewer */}
        <div className="md:col-span-2">
          <LogViewer
            logs={logs}
            logLevel={logLevel}
            onLogLevelChange={handleLogLevelChange}
            isConnected={wsConnected}
          />
        </div>
      </div>
    </div>
  );
}
