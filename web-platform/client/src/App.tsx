import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { ThemeProvider } from "@/components/ThemeProvider";
import { Navigation } from "@/components/Navigation";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useAuth } from "@/hooks/useAuth";
import { useDevices } from "@/hooks/useDevices";

// Pages
import Home from "@/pages/Home";
import Login from "@/pages/Login";
import Devices from "@/pages/Devices";
import Chat from "@/pages/Chat";
import Admin from "@/pages/Admin";
import NotFound from "@/pages/not-found";

function Router() {
  const { user, login, oidcLogin, logout, isAuthenticated } = useAuth();
  const { connectedDevice, connectDevice, disconnectDevice } = useDevices(isAuthenticated);

  return (
    <div className="min-h-screen bg-background">
      <Navigation
        isAuthenticated={isAuthenticated}
        deviceConnected={!!connectedDevice}
        onLogout={logout}
      />
      
      <main>
        <Switch>
          <Route path="/" component={Home} />
          <Route path="/login">
            <Login onLogin={login} onOIDCLogin={oidcLogin} />
          </Route>
          <Route path="/devices">
            {isAuthenticated ? (
              <Devices
                onDeviceConnect={connectDevice}
                onDeviceDisconnect={disconnectDevice}
              />
            ) : (
              <Login onLogin={login} onOIDCLogin={oidcLogin} />
            )}
          </Route>
          <Route path="/chat">
            {isAuthenticated ? (
              <Chat deviceConnected={!!connectedDevice} connectedDeviceId={connectedDevice} />
            ) : (
              <Login onLogin={login} onOIDCLogin={oidcLogin} />
            )}
          </Route>
          <Route path="/admin">
            {isAuthenticated ? (
              <Admin />
            ) : (
              <Login onLogin={login} onOIDCLogin={oidcLogin} />
            )}
          </Route>
          <Route component={NotFound} />
        </Switch>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <ThemeProvider defaultTheme="system" storageKey="ir-theme">
          <Router />
          <Toaster />
        </ThemeProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;