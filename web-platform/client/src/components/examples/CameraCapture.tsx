import { CameraCapture } from '../CameraCapture';
import { ThemeProvider } from '../ThemeProvider';

export default function CameraCaptureExample() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-background p-4">
        <CameraCapture 
          onCapture={(file) => console.log('Captured:', file.name)}
          onClose={() => console.log('Camera closed')}
        />
      </div>
    </ThemeProvider>
  );
}