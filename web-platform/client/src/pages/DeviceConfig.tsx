import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useRoute } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { queryClient } from "@/lib/queryClient";
import { ArrowLeft, Save, Loader2 } from "lucide-react";
import { Link } from "wouter";

export default function DeviceConfig() {
  const [, params] = useRoute("/admin/device/:deviceId/config");
  const deviceId = params?.deviceId;
  const [configText, setConfigText] = useState("");
  const [hasChanges, setHasChanges] = useState(false);
  const { toast } = useToast();

  // Fetch device config
  const { data: config, isLoading } = useQuery({
    queryKey: [`/api/config/device/${deviceId}`],
    enabled: !!deviceId,
  });

  // Load config into editor when data arrives
  useEffect(() => {
    if (config) {
      setConfigText(JSON.stringify(config.config, null, 2));
      setHasChanges(false);
    }
  }, [config]);

  // Update config mutation
  const updateMutation = useMutation({
    mutationFn: async (newConfig: any) => {
      const response = await fetch(`/api/config/device/${deviceId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          config: newConfig,
          changeDescription: "Updated via admin UI",
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Failed to update configuration");
      }

      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/config/device/${deviceId}`] });
      setHasChanges(false);
      toast({
        title: "Success",
        description: "Configuration updated and pushed to device",
      });
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
    try {
      const parsed = JSON.parse(configText);
      updateMutation.mutate(parsed);
    } catch (err) {
      toast({
        title: "Invalid JSON",
        description: "Please fix JSON syntax errors before saving",
        variant: "destructive",
      });
    }
  };

  const handleChange = (value: string) => {
    setConfigText(value);
    setHasChanges(true);
  };

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center gap-4 mb-6">
        <Link href="/admin">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-3xl font-bold">Device Configuration</h1>
          <p className="text-muted-foreground">
            Device: {deviceId}
            {config && ` • Version ${config.version}`}
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Configuration Editor</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="config-editor">
              JSON Configuration
            </Label>
            <Textarea
              id="config-editor"
              value={configText}
              onChange={(e) => handleChange(e.target.value)}
              className="font-mono text-sm h-96"
              placeholder="Loading configuration..."
            />
          </div>

          <div className="flex items-center gap-4">
            <Button
              onClick={handleSave}
              disabled={!hasChanges || updateMutation.isPending}
              className="gap-2"
            >
              {updateMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Save className="h-4 w-4" />
              )}
              Save & Push to Device
            </Button>

            {hasChanges && (
              <span className="text-sm text-muted-foreground">
                Unsaved changes
              </span>
            )}
          </div>

          <div className="text-sm text-muted-foreground space-y-1">
            <p>• Changes will be saved to the database and pushed to the device via MQTT</p>
            <p>• Device will validate and apply the new configuration</p>
            <p>• Configuration history is maintained for rollback</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
