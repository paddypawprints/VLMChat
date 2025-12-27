import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { queryClient } from "@/lib/queryClient";
import { admin } from "@/lib/api";
import { 
  Plus, 
  Trash2, 
  Edit, 
  Server,
  Monitor,
  Cpu,
  HardDrive,
  Loader2
} from "lucide-react";

interface Device {
  id: string;
  name: string;
  type: string;
  ip: string;
  status: string;
  specs?: {
    cpu: string;
    memory: string;
    temperature: number;
    usage: number;
  };
  user_id?: string;
  last_seen?: string;
  created_at?: string;
}

interface AdminService {
  id: string;
  name: string;
  type: string;
  endpoint?: string;
  status: string;
  config?: Record<string, any>;
  created_at?: string;
}

export default function Admin() {
  const [showAddDevice, setShowAddDevice] = useState(false);
  const [showAddService, setShowAddService] = useState(false);
  const [newDevice, setNewDevice] = useState({
    id: "",
    name: "",
    type: "raspberry-pi",
    ip: ""
  });
  const [newService, setNewService] = useState({
    name: "",
    type: "monitoring",
    endpoint: "",
    status: "active"
  });

  const { toast } = useToast();

  // Fetch devices and services using react-query
  const { 
    data: devices = [], 
    isLoading: devicesLoading, 
    error: devicesError 
  } = useQuery<Device[]>({
    queryKey: ['/admin/devices'],
    retry: 2
  });

  const { 
    data: services = [], 
    isLoading: servicesLoading, 
    error: servicesError 
  } = useQuery<AdminService[]>({
    queryKey: ['/admin/services'],
    retry: 2
  });

  // Device mutations
  const createDeviceMutation = useMutation({
    mutationFn: admin.devices.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/admin/devices'] });
      setNewDevice({ id: "", name: "", type: "raspberry-pi", ip: "" });
      setShowAddDevice(false);
      toast({
        title: "Success",
        description: "Device added successfully",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    }
  });

  const deleteDeviceMutation = useMutation({
    mutationFn: admin.devices.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/admin/devices'] });
      toast({
        title: "Success",
        description: "Device deleted successfully",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error", 
        description: error.message,
        variant: "destructive",
      });
    }
  });

  // Service mutations
  const createServiceMutation = useMutation({
    mutationFn: admin.services.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/admin/services'] });
      setNewService({ name: "", type: "monitoring", endpoint: "", status: "active" });
      setShowAddService(false);
      toast({
        title: "Success",
        description: "Service added successfully",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    }
  });

  const deleteServiceMutation = useMutation({
    mutationFn: admin.services.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/admin/services'] });
      toast({
        title: "Success",
        description: "Service deleted successfully",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    }
  });

  const handleAddDevice = () => {
    if (!newDevice.id || !newDevice.name || !newDevice.ip) {
      toast({
        title: "Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    createDeviceMutation.mutate(newDevice);
  };

  const handleAddService = () => {
    if (!newService.name || !newService.type) {
      toast({
        title: "Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    createServiceMutation.mutate(newService);
  };

  const handleDeleteDevice = (deviceId: string) => {
    deleteDeviceMutation.mutate(deviceId);
  };

  const handleDeleteService = (serviceId: string) => {
    deleteServiceMutation.mutate(serviceId);
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'raspberry-pi': return '🔴';
      case 'jetson': return '🟢';
      case 'coral': return '🟡';
      default: return '⚪';
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Admin Panel</h1>
        <p className="text-muted-foreground">
          Manage edge devices and backend services for Independent Research platform.
        </p>
      </div>

      {/* Device Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Device Management
            </CardTitle>
            <Button 
              onClick={() => setShowAddDevice(!showAddDevice)}
              className="gap-2"
              data-testid="button-add-device"
            >
              <Plus className="h-4 w-4" />
              Add Device
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {showAddDevice && (
            <Card className="bg-muted/50">
              <CardContent className="p-4 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="device-id">Device ID *</Label>
                    <Input
                      id="device-id"
                      value={newDevice.id}
                      onChange={(e) => setNewDevice(prev => ({ ...prev, id: e.target.value }))}
                      placeholder="e.g., rpi-002"
                      data-testid="input-device-id"
                    />
                  </div>
                  <div>
                    <Label htmlFor="device-name">Device Name *</Label>
                    <Input
                      id="device-name"
                      value={newDevice.name}
                      onChange={(e) => setNewDevice(prev => ({ ...prev, name: e.target.value }))}
                      placeholder="e.g., Raspberry Pi 4B"
                      data-testid="input-device-name"
                    />
                  </div>
                  <div>
                    <Label htmlFor="device-type">Device Type *</Label>
                    <Select value={newDevice.type} onValueChange={(value) => setNewDevice(prev => ({ ...prev, type: value }))}>
                      <SelectTrigger data-testid="select-device-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="raspberry-pi">Raspberry Pi</SelectItem>
                        <SelectItem value="jetson">NVIDIA Jetson</SelectItem>
                        <SelectItem value="coral">Google Coral</SelectItem>
                        <SelectItem value="ncs">Intel NCS</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="device-ip">IP Address *</Label>
                    <Input
                      id="device-ip"
                      value={newDevice.ip}
                      onChange={(e) => setNewDevice(prev => ({ ...prev, ip: e.target.value }))}
                      placeholder="e.g., 192.168.1.100"
                      data-testid="input-device-ip"
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button 
                    onClick={handleAddDevice} 
                    disabled={createDeviceMutation.isPending}
                    data-testid="button-save-device"
                  >
                    {createDeviceMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                    Add Device
                  </Button>
                  <Button variant="outline" onClick={() => setShowAddDevice(false)} data-testid="button-cancel-device">
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid gap-4">
            {devicesLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <Card key={i} className="bg-card">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <Skeleton className="h-8 w-8 rounded" />
                          <div className="space-y-2">
                            <Skeleton className="h-4 w-[200px]" />
                            <Skeleton className="h-3 w-[300px]" />
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Skeleton className="h-5 w-16" />
                          <Skeleton className="h-8 w-8" />
                          <Skeleton className="h-8 w-8" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : devicesError ? (
              <Card className="bg-card">
                <CardContent className="p-4">
                  <p className="text-destructive">Error loading devices: {devicesError.message}</p>
                </CardContent>
              </Card>
            ) : devices.length === 0 ? (
              <Card className="bg-card">
                <CardContent className="p-4">
                  <p className="text-muted-foreground">No devices found. Add a device to get started.</p>
                </CardContent>
              </Card>
            ) : (
              devices.map((device) => (
                <Card key={device.id} className="bg-card">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <span className="text-2xl">{getDeviceIcon(device.type)}</span>
                        <div>
                          <h3 className="font-semibold">{device.name}</h3>
                          <p className="text-sm text-muted-foreground">
                            ID: {device.id} • IP: {device.ip} • Type: {device.type}
                          </p>
                          {device.specs && (
                            <p className="text-xs text-muted-foreground mt-1">
                              {device.specs.cpu} • {device.specs.memory} • {device.specs.temperature}°C • {device.specs.usage}% usage
                            </p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={device.status === 'connected' ? 'default' : 'secondary'}>
                          {device.status}
                        </Badge>
                        <Button variant="ghost" size="icon" data-testid={`button-edit-device-${device.id}`}>
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          disabled={deleteDeviceMutation.isPending}
                          onClick={() => handleDeleteDevice(device.id)}
                          data-testid={`button-delete-device-${device.id}`}
                        >
                          {deleteDeviceMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      <Separator />

      {/* Service Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Service Management
            </CardTitle>
            <Button 
              onClick={() => setShowAddService(!showAddService)}
              className="gap-2"
              data-testid="button-add-service"
            >
              <Plus className="h-4 w-4" />
              Add Service
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {showAddService && (
            <Card className="bg-muted/50">
              <CardContent className="p-4 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="service-name">Service Name *</Label>
                    <Input
                      id="service-name"
                      value={newService.name}
                      onChange={(e) => setNewService(prev => ({ ...prev, name: e.target.value }))}
                      placeholder="e.g., Model Server"
                      data-testid="input-service-name"
                    />
                  </div>
                  <div>
                    <Label htmlFor="service-type">Service Type *</Label>
                    <Select value={newService.type} onValueChange={(value) => setNewService(prev => ({ ...prev, type: value }))}>
                      <SelectTrigger data-testid="select-service-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="monitoring">Monitoring</SelectItem>
                        <SelectItem value="inference">Inference</SelectItem>
                        <SelectItem value="storage">Storage</SelectItem>
                        <SelectItem value="gateway">Gateway</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="service-endpoint">Endpoint</Label>
                    <Input
                      id="service-endpoint"
                      value={newService.endpoint}
                      onChange={(e) => setNewService(prev => ({ ...prev, endpoint: e.target.value }))}
                      placeholder="e.g., http://localhost:8080"
                      data-testid="input-service-endpoint"
                    />
                  </div>
                  <div>
                    <Label htmlFor="service-status">Status</Label>
                    <Select value={newService.status} onValueChange={(value) => setNewService(prev => ({ ...prev, status: value }))}>
                      <SelectTrigger data-testid="select-service-status">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="active">Active</SelectItem>
                        <SelectItem value="inactive">Inactive</SelectItem>
                        <SelectItem value="maintenance">Maintenance</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button 
                    onClick={handleAddService} 
                    disabled={createServiceMutation.isPending}
                    data-testid="button-save-service"
                  >
                    {createServiceMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                    Add Service
                  </Button>
                  <Button variant="outline" onClick={() => setShowAddService(false)} data-testid="button-cancel-service">
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid gap-4">
            {servicesLoading ? (
              <div className="space-y-4">
                {[1, 2].map((i) => (
                  <Card key={i} className="bg-card">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <Skeleton className="h-8 w-8 rounded" />
                          <div className="space-y-2">
                            <Skeleton className="h-4 w-[200px]" />
                            <Skeleton className="h-3 w-[300px]" />
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Skeleton className="h-5 w-16" />
                          <Skeleton className="h-8 w-8" />
                          <Skeleton className="h-8 w-8" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : servicesError ? (
              <Card className="bg-card">
                <CardContent className="p-4">
                  <p className="text-destructive">Error loading services: {servicesError.message}</p>
                </CardContent>
              </Card>
            ) : services.length === 0 ? (
              <Card className="bg-card">
                <CardContent className="p-4">
                  <p className="text-muted-foreground">No services found. Add a service to get started.</p>
                </CardContent>
              </Card>
            ) : (
              services.map((service) => (
                <Card key={service.id} className="bg-card">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="p-2 bg-muted rounded-lg">
                          {service.type === 'monitoring' ? <Monitor className="h-5 w-5" /> : 
                           service.type === 'inference' ? <Cpu className="h-5 w-5" /> :
                           <HardDrive className="h-5 w-5" />}
                        </div>
                        <div>
                          <h3 className="font-semibold">{service.name}</h3>
                          <p className="text-sm text-muted-foreground">
                            Type: {service.type} {service.endpoint && `• Endpoint: ${service.endpoint}`}
                          </p>
                          {service.config && (
                            <p className="text-xs text-muted-foreground mt-1">
                              Config: {JSON.stringify(service.config)}
                            </p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={service.status === 'active' ? 'default' : 'secondary'}>
                          {service.status}
                        </Badge>
                        <Button variant="ghost" size="icon" data-testid={`button-edit-service-${service.id}`}>
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          disabled={deleteServiceMutation.isPending}
                          onClick={() => handleDeleteService(service.id)}
                          data-testid={`button-delete-service-${service.id}`}
                        >
                          {deleteServiceMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}