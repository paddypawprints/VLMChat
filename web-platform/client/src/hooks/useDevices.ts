import { useState, useEffect } from 'react';
import { devices } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface DeviceSpecs {
  cpu?: string;
  memory?: string;
  temperature?: number;
  usage?: number;
}

interface Device {
  deviceId: string;
  name: string;
  status: string;
  userId: string | null;
  lastSeen: string | null;
  createdAt: string;
}

export function useDevices(isAuthenticated: boolean) {
  const [deviceList, setDeviceList] = useState<Device[]>([]);
  const [loading, setLoading] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [connectedDevice, setConnectedDevice] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchDevices = async () => {
    if (!isAuthenticated) return;
    
    try {
      setLoading(true);
      // Fetch only active/connected devices
      const data = await devices.list('connected');
      setDeviceList(data);
      
      // Update connected device state
      const connected = data.find((d: Device) => d.status === 'connected');
      setConnectedDevice(connected?.deviceId || null);
    } catch (error) {
      console.error('Failed to fetch devices:', error);
      toast({
        title: "Error",
        description: "Failed to fetch devices",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const connectDevice = async (deviceId: string) => {
    try {
      setLoading(true);
      await devices.connect(deviceId);
      setConnectedDevice(deviceId);
      
      // Update device status locally
      setDeviceList(prev => prev.map(device => 
        device.deviceId === deviceId 
          ? { ...device, status: 'connected' }
          : { ...device, status: 'disconnected' }
      ));

      toast({
        title: "Success",
        description: "Device connected successfully",
      });
    } catch (error) {
      console.error('Failed to connect device:', error);
      toast({
        title: "Error",
        description: "Failed to connect device",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const disconnectDevice = async (deviceId: string) => {
    try {
      setLoading(true);
      await devices.disconnect(deviceId);
      setConnectedDevice(null);
      
      // Update device status locally
      setDeviceList(prev => prev.map(device => 
        device.deviceId === deviceId 
          ? { ...device, status: 'disconnected' }
          : device
      ));

      toast({
        title: "Success",
        description: "Device disconnected",
      });
    } catch (error) {
      console.error('Failed to disconnect device:', error);
      toast({
        title: "Error",
        description: "Failed to disconnect device",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const scanDevices = async () => {
    if (!isAuthenticated) return;

    try {
      setScanning(true);
      const result = await devices.scan();
      
      toast({
        title: "Scan Complete",
        description: result.message,
      });
      
      // Refresh device list after scan
      await fetchDevices();
    } catch (error) {
      console.error('Failed to scan devices:', error);
      toast({
        title: "Error",
        description: "Device scan failed",
        variant: "destructive",
      });
    } finally {
      setScanning(false);
    }
  };

  useEffect(() => {
    if (!isAuthenticated) return;
    
    fetchDevices();
    
    // Auto-refresh every 10 seconds to show updated heartbeat status
    const interval = setInterval(() => {
      fetchDevices();
    }, 10000);
    
    return () => clearInterval(interval);
  }, [isAuthenticated]);

  return {
    devices: deviceList,
    loading,
    scanning,
    connectedDevice,
    connectDevice,
    disconnectDevice,
    scanDevices,
    refreshDevices: fetchDevices,
  };
}