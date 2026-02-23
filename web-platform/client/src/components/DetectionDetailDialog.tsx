import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { VlmStatusBadge } from "@/components/VlmStatusBadge";
import { ChevronLeft, ChevronRight, Camera, Clock, Target } from "lucide-react";

interface Detection {
  id: string;
  deviceId: string;
  timestamp: string;
  bbox: [number, number, number, number];
  confidence: number;
  category: string;
  categoryId: number;
  attributes?: Record<string, { value: boolean; confidence: number }>;
  imageUrl: string;
  metadata?: Record<string, any>;
  vlmVerified?: boolean;
  vlmRequired?: boolean;
  vlmTimeout?: boolean;
  vlmError?: string;
  vlmResult?: string;
}

interface DetectionDetailDialogProps {
  detection: Detection;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: (direction: "prev" | "next") => void;
}

export function DetectionDetailDialog({
  detection,
  open,
  onOpenChange,
  onNavigate,
}: DetectionDetailDialogProps) {
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const formatAttribute = (key: string) => {
    // Convert camelCase to Title Case
    return key.replace(/([A-Z])/g, " $1").replace(/^./, (str) => str.toUpperCase());
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Detection Details</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Image */}
          <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
            <img
              src={detection.imageUrl}
              alt="Detection"
              className="w-full h-full object-contain"
            />
          </div>

          {/* Metadata */}
          <div className="grid gap-4">
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2 flex-1">
                <Camera className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">Camera:</span>
                <span className="text-muted-foreground">{detection.deviceId}</span>
              </div>
              <div className="flex items-center gap-2 flex-1">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">Time:</span>
                <span className="text-muted-foreground">
                  {formatTimestamp(detection.timestamp)}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">Confidence:</span>
                <Badge variant="secondary">
                  {Math.round(detection.confidence * 100)}%
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium">Category:</span>
                <Badge>{detection.category}</Badge>
              </div>
              {/* VLM Status */}
              <div className="flex items-center gap-2">
                <VlmStatusBadge
                  vlmRequired={detection.vlmRequired}
                  vlmStatus={detection.vlmStatus}
                  vlmResponse={detection.vlmResponse}
                />
              </div>
            </div>
          </div>

          {/* Attributes */}
          {detection.attributes && Object.keys(detection.attributes).length > 0 && (
            <>
              <Separator />
              <div>
                <h4 className="text-sm font-semibold mb-3">Attributes</h4>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(detection.attributes).map(([key, attr]) => (
                    <div
                      key={key}
                      className="flex items-center justify-between text-sm"
                    >
                      <span className="text-muted-foreground">
                        {formatAttribute(key)}:
                      </span>
                      <div className="flex items-center gap-2">
                        <span
                          className={
                            attr.value ? "text-green-600" : "text-muted-foreground"
                          }
                        >
                          {attr.value ? "Yes" : "No"}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          ({Math.round(attr.confidence * 100)}%)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* VLM Verification Details */}
          {(detection.vlmResult || detection.vlmError) && (
            <>
              <Separator />
              <div>
                <h4 className="text-sm font-semibold mb-3">VLM Verification</h4>
                {detection.vlmResult && (
                  <p className="text-sm text-muted-foreground">
                    {detection.vlmResult}
                  </p>
                )}
                {detection.vlmError && (
                  <p className="text-sm text-yellow-600 dark:text-yellow-400">
                    ⚠ {detection.vlmError}
                  </p>
                )}
              </div>
            </>
          )}
        </div>

        <DialogFooter className="flex justify-between items-center">
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => onNavigate("prev")}>
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </Button>
            <Button variant="outline" size="sm" onClick={() => onNavigate("next")}>
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
