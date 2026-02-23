import { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { Plus, Trash2, Camera, Clock, AlertCircle, Loader2, Info, Pencil } from "lucide-react";
import { DetectionDetailDialog } from "@/components/DetectionDetailDialog";
import { SearchTermFilterDialog } from "@/components/SearchTermFilterDialog";

interface SearchTerm {
  id: string;
  search_string: string;
  created_at: string;
  category_mask: boolean[];
  category_colors: (string | null)[];
  attribute_mask: boolean[];
  attribute_colors: (string | null)[];
  groq_response?: any;
}

interface Detection {
  id: string;
  searchTermId: string;
  deviceId: string;
  timestamp: string;
  bbox: [number, number, number, number];
  confidence: number;
  category: string;
  categoryId: number;
  attributes?: Record<string, { value: boolean; confidence: number }>;
  imageUrl: string;
  metadata?: Record<string, any>;
}

export default function Search() {
  const [newSearchString, setNewSearchString] = useState("");
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);
  const [selectedFilter, setSelectedFilter] = useState<SearchTerm | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Fetch search terms
  const { data: searchTerms = [], isLoading: termsLoading } = useQuery<SearchTerm[]>({
    queryKey: ["/api/search-terms"],
  });

  // Fetch devices to connect WebSockets
  const { data: devices = [] } = useQuery<Array<{ id: string; name: string }>>({
    queryKey: ["/api/devices"],
  });

  const handleAlert = useCallback((deviceId: string, alertData: any) => {
    console.log('[Search] Alert received from', deviceId, alertData);
    
    // Convert alert to detection format
    const detection: Detection = {
      id: `${deviceId}-${Date.now()}`,
      searchTermId: alertData.watchlist_item_id || '',
      deviceId: deviceId,
      timestamp: alertData.timestamp,
      bbox: alertData.metadata?.bounding_box 
        ? [
            alertData.metadata.bounding_box.x,
            alertData.metadata.bounding_box.y,
            alertData.metadata.bounding_box.x + alertData.metadata.bounding_box.width,
            alertData.metadata.bounding_box.y + alertData.metadata.bounding_box.height,
          ]
        : [0, 0, 0, 0],
      confidence: alertData.confidence || 0,
      category: alertData.metadata?.category || 'unknown',
      categoryId: alertData.metadata?.category_id || 0,
      attributes: alertData.metadata?.attributes,
      imageUrl: alertData.image_url || `data:image/jpeg;base64,${alertData.image}`,
      metadata: alertData.metadata,
    };

    // Add to detections list (most recent first)
    setDetections(prev => [detection, ...prev].slice(0, 200)); // Keep last 200
  }, []);

  // Connect to WebSocket for all devices to receive alerts
  useEffect(() => {
    if (devices.length === 0) return;

    const wsConnections: WebSocket[] = [];

    devices.forEach((device) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws?deviceId=${device.id}`;
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log(`[Search] WebSocket connected for device ${device.id}`);
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'alert') {
            handleAlert(device.id, message.message);
          }
        } catch (error) {
          console.error('[Search] Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error(`[Search] WebSocket error for device ${device.id}:`, error);
      };
      
      ws.onclose = () => {
        console.log(`[Search] WebSocket closed for device ${device.id}`);
      };
      
      wsConnections.push(ws);
    });

    // Cleanup on unmount
    return () => {
      wsConnections.forEach(ws => ws.close());
    };
  }, [devices, handleAlert]);

  const detectionsLoading = false; // No API loading since we're using WebSocket

  const addMutation = useMutation({
    mutationFn: async (searchString: string) => {
      const response = await apiRequest("POST", "/api/search-terms", { searchString });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/search-terms"] });
      queryClient.invalidateQueries({ queryKey: ["/api/search-terms/detections"] });
      toast({
        title: "Search term added",
        description: "Filter parsed and updated on all cameras",
      });
      setNewSearchString("");
    },
    onError: (error: Error) => {
      toast({
        title: "Error adding search term",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const response = await apiRequest("DELETE", `/api/search-terms/${id}`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/search-terms"] });
      queryClient.invalidateQueries({ queryKey: ["/api/search-terms/detections"] });
      toast({
        title: "Search term removed",
        description: "Filter updated on all cameras",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to delete search term",
        variant: "destructive",
      });
    },
  });

  const handleAdd = (e: React.FormEvent) => {
    e.preventDefault();
    if (newSearchString.trim()) {
      addMutation.mutate(newSearchString.trim());
    }
  };

  const handleDelete = (id: string, searchString: string) => {
    if (confirm(`Remove "${searchString}" from search?`)) {
      deleteMutation.mutate(id);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  const totalDetections = detections.length;
  const deviceCount = new Set(detections.map((d) => d.deviceId)).size;

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          🔍 Search & Detections
        </h1>
        <p className="text-muted-foreground mt-2">
          Add search terms to detect objects across all cameras
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Active Search Terms</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleAdd} className="flex gap-2">
                <Input
                  placeholder="e.g., person wearing green shirt"
                  value={newSearchString}
                  onChange={(e) => setNewSearchString(e.target.value)}
                  disabled={addMutation.isPending}
                />
                <Button
                  type="submit"
                  size="sm"
                  disabled={!newSearchString.trim() || addMutation.isPending}
                >
                  {addMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Plus className="h-4 w-4" />
                  )}
                </Button>
              </form>

              {termsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-10 w-full" />
                  ))}
                </div>
              ) : searchTerms.length > 0 ? (
                <div className="space-y-2">
                  {searchTerms.map((term) => (
                    <div
                      key={term.id}
                      className="flex items-center justify-between p-2 rounded-md bg-muted"
                    >
                      <span className="text-sm flex-1">{term.search_string}</span>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedFilter(term)}
                          title="View filter details"
                        >
                          <Info className="h-4 w-4 text-blue-500" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(term.id, term.search_string)}
                          disabled={deleteMutation.isPending}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No search terms yet</p>
                </div>
              )}

              <div className="pt-4 border-t">
                <h4 className="text-xs font-medium mb-2 text-muted-foreground">
                  EXAMPLES
                </h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• red car</li>
                  <li>• person with backpack</li>
                  <li>• female wearing blue hat</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Detections</CardTitle>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Camera className="h-4 w-4" />
                    {totalDetections} detection{totalDetections !== 1 ? "s" : ""}
                  </span>
                  <span>
                    {deviceCount} camera{deviceCount !== 1 ? "s" : ""}
                  </span>
                </div>
              </div>
            </CardHeader>
            <CardContent className="h-[600px] overflow-hidden">
              {detectionsLoading ? (
                <div className="grid grid-flow-col auto-cols-max gap-4 h-full overflow-x-auto overflow-y-hidden py-4">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="h-[550px]">
                      <Skeleton className="h-full w-[300px]" />
                    </div>
                  ))}
                </div>
              ) : detections.length > 0 ? (
                <div className="grid grid-flow-col auto-cols-max gap-6 h-full overflow-x-auto overflow-y-hidden py-4">
                  {detections.map((detection) => (
                    <div
                      key={detection.id}
                      className="relative cursor-pointer h-[550px]"
                      onClick={() => setSelectedImage(detection.imageUrl)}
                    >
                      <img
                        src={detection.imageUrl}
                        alt={`Detection ${formatTimestamp(detection.timestamp)}`}
                        className="h-full w-auto rounded-lg hover:opacity-90 transition-opacity"
                        loading="lazy"
                      />
                      <Badge
                        variant="secondary"
                        className="absolute top-2 right-2 bg-background/90 backdrop-blur"
                      >
                        {Math.round(detection.confidence * 100)}%
                      </Badge>
                      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-background/80 backdrop-blur rounded px-3 py-2 space-y-1 text-center min-w-[200px]">
                        <div className="flex items-center justify-center gap-1 text-xs text-foreground">
                          <Clock className="h-3 w-3" />
                          {formatTimestamp(detection.timestamp)}
                        </div>
                        <div className="flex items-center justify-center gap-1 text-xs text-foreground">
                          <Camera className="h-3 w-3" />
                          {detection.deviceId.split("-")[0]}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-16">
                  <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-semibold mb-2">No detections yet</h3>
                  <p className="text-muted-foreground text-sm">
                    Add search terms above to start detecting objects
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {selectedDetection && (
        <DetectionDetailDialog
          detection={selectedDetection}
          open={!!selectedDetection}
          onOpenChange={(open) => !open && setSelectedDetection(null)}
          onNavigate={(direction) => {
            const currentIndex = detections.findIndex((d) => d.id === selectedDetection.id);
            if (direction === "prev" && currentIndex > 0) {
              setSelectedDetection(detections[currentIndex - 1]);
            } else if (direction === "next" && currentIndex < detections.length - 1) {
              setSelectedDetection(detections[currentIndex + 1]);
            }
          }}
        />
      )}

      {/* Full-size image modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setSelectedImage(null)}
        >
          <img
            src={selectedImage}
            alt="Full size detection"
            className="max-w-full max-h-full object-contain"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      <SearchTermFilterDialog
        filter={selectedFilter}
        open={!!selectedFilter}
        onOpenChange={(open) => !open && setSelectedFilter(null)}
      />
    </div>
  );
}
