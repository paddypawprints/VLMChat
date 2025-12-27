import { useState } from 'react';
import { ChatInterface } from '../ChatInterface';
import { ThemeProvider } from '../ThemeProvider';

export default function ChatInterfaceExample() {
  const [debugMode, setDebugMode] = useState(false);

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-background p-4">
        <div className="max-w-4xl mx-auto h-[80vh] border rounded-lg bg-card">
          <ChatInterface 
            deviceConnected={true}
            debugMode={debugMode}
            onToggleDebug={() => setDebugMode(!debugMode)}
            onSendMessage={(message, images) => console.log('Send:', { message, imageCount: images?.length })}
            onTakePhoto={() => console.log('Take photo')}
          />
        </div>
      </div>
    </ThemeProvider>
  );
}