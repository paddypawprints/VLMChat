import { useState } from "react";
import { ChatInterface } from "@/components/ChatInterface";
import { CameraCapture } from "@/components/CameraCapture";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";

interface ChatProps {
  deviceConnected?: boolean;
  connectedDeviceId?: string | null;
}

export default function Chat({ deviceConnected = false, connectedDeviceId }: ChatProps) {
  const [debugMode, setDebugMode] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [capturedImage, setCapturedImage] = useState<File | null>(null);

  const handleTakePhoto = () => {
    setShowCamera(true);
  };

  const handleCameraCapture = (imageFile: File) => {
    setCapturedImage(imageFile);
    setShowCamera(false);
  };

  return (
    <div className="container mx-auto p-6 h-screen flex flex-col">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">AI Chat</h1>
        <p className="text-muted-foreground">
          Chat with your AI models running on connected edge devices.
        </p>
      </div>
      
      <div className="flex-1 bg-card rounded-lg border overflow-hidden">
        <ChatInterface
          deviceConnected={deviceConnected}
          deviceId={connectedDeviceId}
          debugMode={debugMode}
          onToggleDebug={() => setDebugMode(!debugMode)}
          onTakePhoto={handleTakePhoto}
          capturedImage={capturedImage}
          onClearCapturedImage={() => setCapturedImage(null)}
        />
      </div>

      <Dialog open={showCamera} onOpenChange={setShowCamera}>
        <DialogContent className="max-w-3xl">
          <VisuallyHidden>
            <DialogTitle>Camera Capture</DialogTitle>
          </VisuallyHidden>
          <CameraCapture
            onCapture={handleCameraCapture}
            onClose={() => setShowCamera(false)}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}