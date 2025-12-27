import { DeviceConnection } from '../DeviceConnection';
import { ThemeProvider } from '../ThemeProvider';

export default function DeviceConnectionExample() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-background p-6">
        <DeviceConnection 
          onConnect={(deviceId) => console.log('Connect to:', deviceId)}
          onDisconnect={(deviceId) => console.log('Disconnect from:', deviceId)}
        />
      </div>
    </ThemeProvider>
  );
}