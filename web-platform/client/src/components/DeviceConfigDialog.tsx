import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { JsonViewer } from "@textea/json-viewer";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { queryClient } from "@/lib/queryClient";
import { Loader2 } from "lucide-react";

interface DeviceConfigDialogProps {
  deviceId: string;
  deviceName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// Helper to check if value is an object
function isObject(value: any): boolean {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// Helper to update value at path in object
function updateAtPath(obj: any, path: (string | number)[], value: any): any {
  if (path.length === 0) return value;
  
  const [head, ...tail] = path;
  const newObj = Array.isArray(obj) ? [...obj] : { ...obj };
  newObj[head] = updateAtPath(newObj[head], tail, value);
  return newObj;
}

export function DeviceConfigDialog({ deviceId, deviceName, open, onOpenChange }: DeviceConfigDialogProps) {
  const { toast } = useToast();
  const [configData, setConfigData] = useState<any>(null);

  // Fetch current device config
  const { data: config, isLoading } = useQuery({
    queryKey: [`/config/device/${deviceId}`],
    queryFn: async () => {
      const res = await fetch(`/api/config/device/${deviceId}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('ir-session')}`,
        },
      });
      if (!res.ok) throw new Error('Failed to fetch config');
      const data = await res.json();
      setConfigData(data.config);
      return data;
    },
    enabled: open,
  });

  // Update device config mutation
  const updateMutation = useMutation({
    mutationFn: async (newConfig: any) => {
      const res = await fetch(`/api/config/device/${deviceId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('ir-session')}`,
        },
        body: JSON.stringify({ config: newConfig }),
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || 'Failed to update config');
      }
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/config/device/${deviceId}`] });
      toast({
        title: "Success",
        description: "Device configuration updated and published to device",
      });
      onOpenChange(false);
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleSave = () => {
    if (configData) {
      updateMutation.mutate(configData);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure {deviceName}</DialogTitle>
          <DialogDescription>
            Edit device configuration (tasks and sinks). Only values can be edited - structure is read-only. 
            Changes will be published to THIS device only via MQTT.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto border rounded-lg p-4 bg-muted/30">
          {isLoading ? (
            <div className="flex items-center justify-center p-8">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : configData ? (
            <JsonViewer
              value={configData}
              theme="auto"
              defaultInspectDepth={10}
              editable={true}
              onEdit={(params) => {
                const { path, newValue, oldValue } = params;
                
                // Only allow editing leaf nodes (primitives)
                const isLeaf = !isObject(newValue) && !isObject(oldValue);
                
                if (!isLeaf) {
                  toast({
                    title: "Cannot edit structure",
                    description: "Only values can be edited, not object keys or structure",
                    variant: "destructive",
                  });
                  return;
                }
                
                // Update config at path
                const updated = updateAtPath(configData, path, newValue);
                setConfigData(updated);
              }}
              onAdd={() => {
                toast({
                  title: "Cannot add fields",
                  description: "Structure is read-only",
                  variant: "destructive",
                });
              }}
              onDelete={() => {
                toast({
                  title: "Cannot delete fields",
                  description: "Structure is read-only",
                  variant: "destructive",
                });
              }}
              displayDataTypes={false}
              displaySize={false}
              enableClipboard={false}
            />
          ) : (
            <div className="text-sm text-muted-foreground p-4">
              No configuration found
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleSave}
            disabled={updateMutation.isPending || isLoading || !configData}
          >
            {updateMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Save & Publish
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
