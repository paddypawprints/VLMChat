import { Navigation } from '../Navigation';
import { ThemeProvider } from '../ThemeProvider';

export default function NavigationExample() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-background">
        <Navigation 
          isAuthenticated={true}
          deviceConnected={true}
          onLogout={() => console.log('Logout clicked')}
        />
        <div className="p-8">
          <h1 className="text-2xl font-bold">Page Content</h1>
          <p className="text-muted-foreground">This shows how the navigation looks with content.</p>
        </div>
      </div>
    </ThemeProvider>
  );
}