import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useDevices } from "@/hooks/useDevices";
import { useAuth } from "@/hooks/useAuth";
import { 
  Cpu, 
  Wifi, 
  WifiOff, 
  RefreshCw, 
  Zap,
  HardDrive,
  Thermometer,
  Activity
} from "lucide-react";

interface DeviceConnectionProps {
  onConnect?: (deviceId: string) => Promise<void>;
  onDisconnect?: (deviceId: string) => Promise<void>;
}

export function DeviceConnection({ onConnect, onDisconnect }: DeviceConnectionProps) {
  const { isAuthenticated } = useAuth();
  const { 
    devices, 
    loading,
    refreshDevices
  } = useDevices(isAuthenticated);

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'raspberry-pi': return '🔴';
      case 'jetson': return '🟢';
      case 'coral': return '🟡';
      case 'ncs': return '🔵';
      default: return '⚪';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Connected Devices
            </CardTitle>
            <Button 
              onClick={refreshDevices}
              disabled={loading}
              variant="outline"
              size="sm"
              className="gap-2"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Devices connect automatically via MQTT. Configure your edge devices to publish to topics like{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">devices/&lt;device-id&gt;/status</code>
          </p>
        </CardContent>
      </Card>

      {/* Device List */}
      <div className="space-y-4">
        {loading && devices.length === 0 && (
          <Alert>
            <AlertDescription>Loading devices...</AlertDescription>
          </Alert>
        )}
        
        {!loading && devices.length === 0 && (
          <Alert>
            <AlertDescription>
              No devices connected yet. Configure your edge devices to connect via MQTT on port 1883.
            </AlertDescription>
          </Alert>
        )}
        
        {devices.map((device) => (
          <Card key={device.id} className="hover-elevate">
            <CardContent className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="text-2xl">{getDeviceIcon(device.type)}</div>
                  <div>
                    <h4 className="font-semibold flex items-center gap-2">
                      {device.name}
                      <Badge 
                        variant={device.status === 'connected' ? 'default' : 'secondary'}
                        className="gap-1"
                      >
                        {device.status === 'connected' ? (
                          <>
                            <Wifi className="h-3 w-3" />
                            Connected
                          </>
                        ) : (
                          <>
                            <WifiOff className="h-3 w-3" />
                            Disconnected
                          </>
                        )}
                      </Badge>
                    </h4>
                    <p className="text-sm text-muted-foreground">{device.ip}</p>
                    {device.lastSeen && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Last seen: {new Date(device.lastSeen).toLocaleString()}
                      </p>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Device Specs */}
              {device.specs && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">CPU:</span>
                    <span>{device.specs.cpu}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">RAM:</span>
                    <span>{device.specs.memory}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Thermometer className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Temp:</span>
                    <span>{device.specs.temperature}°C</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Usage:</span>
                    <span>{device.specs.usage}%</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}