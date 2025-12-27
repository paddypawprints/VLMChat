import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "./ThemeToggle";
import { Cpu, Wifi, WifiOff } from "lucide-react";
import logoImage from "@/assets/logo.png";

interface NavigationProps {
  isAuthenticated?: boolean;
  deviceConnected?: boolean;
  onLogout?: () => void;
}

export function Navigation({ isAuthenticated = false, deviceConnected = false, onLogout }: NavigationProps) {
  const [location] = useLocation();

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-14 items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" data-testid="link-home">
              <div className="flex items-center gap-2 hover-elevate active-elevate-2 p-2 -m-2 rounded-md transition-colors">
                <img src={logoImage} alt="Independent Research" className="h-8 w-8" />
                <span className="font-semibold text-lg">Independent Research</span>
              </div>
            </Link>
            
            {isAuthenticated && (
              <div className="hidden md:flex items-center gap-4">
                <Link href="/devices" data-testid="link-devices">
                  <Button 
                    variant={location === "/devices" ? "default" : "ghost"}
                    size="sm"
                    className="gap-2"
                  >
                    <Cpu className="h-4 w-4" />
                    Devices
                  </Button>
                </Link>
                <Link href="/chat" data-testid="link-chat">
                  <Button 
                    variant={location === "/chat" ? "default" : "ghost"}
                    size="sm"
                  >
                    Chat
                  </Button>
                </Link>
                <Link href="/admin" data-testid="link-admin">
                  <Button 
                    variant={location === "/admin" ? "default" : "ghost"}
                    size="sm"
                  >
                    Admin
                  </Button>
                </Link>
              </div>
            )}
          </div>

          <div className="flex items-center gap-4">
            {isAuthenticated && (
              <div className="flex items-center gap-3">
                <Badge 
                  variant={deviceConnected ? "default" : "secondary"} 
                  className="gap-1"
                  data-testid="status-device-connection"
                >
                  {deviceConnected ? (
                    <>
                      <Wifi className="h-3 w-3" />
                      Connected
                    </>
                  ) : (
                    <>
                      <WifiOff className="h-3 w-3" />
                      Disconnected
                    </>
                  )}
                </Badge>
              </div>
            )}
            
            <ThemeToggle />
            
            {isAuthenticated ? (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={onLogout}
                data-testid="button-logout"
              >
                Logout
              </Button>
            ) : location !== "/login" && (
              <Link href="/login" data-testid="link-login">
                <Button size="sm">
                  Login
                </Button>
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}