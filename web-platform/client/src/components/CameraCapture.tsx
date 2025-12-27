import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Camera, Square, RotateCcw, Check, X } from "lucide-react";

interface CameraCaptureProps {
  onCapture?: (imageFile: File) => void;
  onClose?: () => void;
}

export function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const [isActive, setIsActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [error, setError] = useState<string>("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = useCallback(async () => {
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment' // Prefer back camera
        }
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsActive(true);
    } catch (err) {
      console.error('Camera access failed:', err);
      setError("Failed to access camera. Please check permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsActive(false);
    setCapturedImage(null);
  }, []);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0);
    
    // Convert to data URL
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    setCapturedImage(dataURL);
    
    console.log('Photo captured');
  }, []);

  const handleSave = useCallback(() => {
    if (!capturedImage) return;
    
    // Convert data URL to File
    const dataURLtoFile = (dataurl: string, filename: string) => {
      const arr = dataurl.split(',');
      const mime = arr[0].match(/:(.*?);/)![1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      return new File([u8arr], filename, { type: mime });
    };
    
    const file = dataURLtoFile(capturedImage, `camera-capture-${Date.now()}.jpg`);
    onCapture?.(file);
    
    // Clean up
    stopCamera();
    onClose?.();
  }, [capturedImage, onCapture, onClose, stopCamera]);

  const handleRetake = useCallback(() => {
    setCapturedImage(null);
  }, []);

  // Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Camera Capture
          </CardTitle>
          <div className="flex items-center gap-2">
            {isActive && (
              <Badge variant="outline" className="gap-1">
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
                Recording
              </Badge>
            )}
            <Button variant="ghost" size="icon" onClick={() => { stopCamera(); onClose?.(); }} data-testid="button-close-camera">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
          {!isActive && !capturedImage && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center space-y-4">
                <Camera className="h-16 w-16 text-muted-foreground mx-auto" />
                <div>
                  <p className="font-medium">Camera Access Required</p>
                  <p className="text-sm text-muted-foreground">
                    Click start to enable camera and capture photos
                  </p>
                </div>
              </div>
            </div>
          )}
          
          {capturedImage && (
            <img 
              src={capturedImage} 
              alt="Captured" 
              className="w-full h-full object-cover"
            />
          )}
          
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${isActive && !capturedImage ? 'block' : 'hidden'}`}
          />
          
          <canvas ref={canvasRef} className="hidden" />
        </div>

        <div className="flex justify-center gap-2">
          {!isActive && !capturedImage && (
            <Button onClick={startCamera} className="gap-2" data-testid="button-start-camera">
              <Camera className="h-4 w-4" />
              Start Camera
            </Button>
          )}
          
          {isActive && !capturedImage && (
            <>
              <Button onClick={capturePhoto} size="lg" className="gap-2" data-testid="button-capture-photo">
                <Square className="h-4 w-4" />
                Capture
              </Button>
              <Button variant="outline" onClick={stopCamera} data-testid="button-stop-camera">
                Stop
              </Button>
            </>
          )}
          
          {capturedImage && (
            <>
              <Button onClick={handleSave} className="gap-2" data-testid="button-save-photo">
                <Check className="h-4 w-4" />
                Use Photo
              </Button>
              <Button variant="outline" onClick={handleRetake} className="gap-2" data-testid="button-retake-photo">
                <RotateCcw className="h-4 w-4" />
                Retake
              </Button>
              <Button variant="ghost" onClick={() => { stopCamera(); onClose?.(); }} data-testid="button-cancel-photo">
                Cancel
              </Button>
            </>
          )}
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Captured photos will be included in your next chat message
        </p>
      </CardContent>
    </Card>
  );
}