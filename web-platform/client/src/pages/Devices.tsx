import { useDevices } from "@/hooks/useDevices";
import { useAuth } from "@/hooks/useAuth";
import { DeviceConnection } from "@/components/DeviceConnection";

interface DevicesProps {
  onDeviceConnect?: (deviceId: string) => Promise<void>;
  onDeviceDisconnect?: (deviceId: string) => Promise<void>;
}

export default function Devices({ onDeviceConnect, onDeviceDisconnect }: DevicesProps) {
  const { isAuthenticated } = useAuth();
  const { devices, loading, scanning, scanDevices } = useDevices(isAuthenticated);

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Edge Devices</h1>
        <p className="text-muted-foreground">
          Connect to your edge devices to start chatting with AI models.
        </p>
      </div>
      
      <DeviceConnection
        onConnect={onDeviceConnect}
        onDisconnect={onDeviceDisconnect}
      />
    </div>
  );
}