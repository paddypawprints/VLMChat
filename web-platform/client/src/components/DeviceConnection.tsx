import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useDevices } from "@/hooks/useDevices";
import { useAuth } from "@/hooks/useAuth";
import { Link } from "wouter";
import { 
  Cpu, 
  Wifi, 
  WifiOff, 
  RefreshCw, 
  Zap,
  HardDrive,
  Thermometer,
  Activity,
  Eye
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
      case 'mac': return '💻';
      default: return '⚪';
    }
  };

  // Check if device is truly online based on heartbeat (within last 60 seconds)
  const isDeviceAlive = (lastSeen: Date | null | undefined) => {
    if (!lastSeen) return false;
    const now = new Date().getTime();
    const lastSeenTime = new Date(lastSeen).getTime();
    const diffSeconds = (now - lastSeenTime) / 1000;
    return diffSeconds < 60; // Consider alive if heartbeat within last 60 seconds
  };

  const getHeartbeatStatus = (lastSeen: Date | null | undefined) => {
    if (!lastSeen) return 'Never';
    const now = new Date().getTime();
    const lastSeenTime = new Date(lastSeen).getTime();
    const diffSeconds = (now - lastSeenTime) / 1000;
    
    if (diffSeconds < 60) return 'Just now';
    if (diffSeconds < 120) return '1 minute ago';
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)} minutes ago`;
    if (diffSeconds < 7200) return '1 hour ago';
    return new Date(lastSeen).toLocaleString();
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
          <Card key={device.deviceId} className="hover-elevate">
            <CardContent className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="text-2xl">💻</div>
                  <div>
                    <h4 className="font-semibold flex items-center gap-2">
                      {device.name}
                      <Badge 
                        variant={device.status === 'connected' ? 'default' : 'secondary'}
                        className="gap-1"
                      >
                        {device.status === 'connected' ? (
                          <>
                            <Activity className="h-3 w-3 animate-pulse" />
                            Online
                          </>
                        ) : (
                          <>
                            <WifiOff className="h-3 w-3" />
                            Offline
                          </>
                        )}
                      </Badge>
                    </h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Last seen: {device.lastSeen ? new Date(device.lastSeen).toLocaleString() : 'Never'}
                    </p>
                  </div>
                </div>
                
                {/* View Details Button */}
                <Link href={`/devices/${device.deviceId}`}>
                  <Button variant="outline" size="sm" className="gap-2">
                    <Eye className="h-4 w-4" />
                    View Details
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}