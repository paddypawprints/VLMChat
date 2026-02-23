import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "./ThemeToggle";
import { Cpu, Wifi, WifiOff, Menu, X } from "lucide-react";
import logoImage from "@/assets/logo.png";
import { useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

interface NavigationProps {
  isAuthenticated?: boolean;
  deviceConnected?: boolean;
  onLogout?: () => void;
}

export function Navigation({ isAuthenticated = false, deviceConnected = false, onLogout }: NavigationProps) {
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

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
            
            <div className="hidden md:flex items-center gap-4">
              <Link href="/technology" data-testid="link-technology">
                <Button 
                  variant={location === "/technology" ? "default" : "ghost"}
                  size="sm"
                >
                  Technology
                </Button>
              </Link>
            </div>
            
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
                <Link href="/search" data-testid="link-search">
                  <Button 
                    variant={location === "/search" ? "default" : "ghost"}
                    size="sm"
                  >
                    🔍 Search
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
            
            {/* Mobile Menu */}
            <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
              <SheetTrigger asChild className="md:hidden">
                <Button variant="ghost" size="sm">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right">
                <SheetHeader>
                  <SheetTitle>Menu</SheetTitle>
                </SheetHeader>
                <div className="flex flex-col gap-4 mt-6">
                  <Link href="/technology" onClick={() => setMobileMenuOpen(false)}>
                    <Button 
                      variant={location === "/technology" ? "default" : "ghost"}
                      className="w-full justify-start"
                    >
                      Technology
                    </Button>
                  </Link>
                  
                  {isAuthenticated && (
                    <>
                      <Link href="/devices" onClick={() => setMobileMenuOpen(false)}>
                        <Button 
                          variant={location === "/devices" ? "default" : "ghost"}
                          className="w-full justify-start gap-2"
                        >
                          <Cpu className="h-4 w-4" />
                          Devices
                        </Button>
                      </Link>
                      <Link href="/search" onClick={() => setMobileMenuOpen(false)}>
                        <Button 
                          variant={location === "/search" ? "default" : "ghost"}
                          className="w-full justify-start"
                        >
                          🔍 Search
                        </Button>
                      </Link>
                      <Link href="/chat" onClick={() => setMobileMenuOpen(false)}>
                        <Button 
                          variant={location === "/chat" ? "default" : "ghost"}
                          className="w-full justify-start"
                        >
                          Chat
                        </Button>
                      </Link>
                      <Link href="/admin" onClick={() => setMobileMenuOpen(false)}>
                        <Button 
                          variant={location === "/admin" ? "default" : "ghost"}
                          className="w-full justify-start"
                        >
                          Admin
                        </Button>
                      </Link>
                    </>
                  )}
                  
                  <div className="border-t pt-4 mt-2">
                    {isAuthenticated ? (
                      <Button 
                        variant="outline" 
                        className="w-full"
                        onClick={() => {
                          setMobileMenuOpen(false);
                          onLogout?.();
                        }}
                      >
                        Logout
                      </Button>
                    ) : location !== "/login" && (
                      <Link href="/login" onClick={() => setMobileMenuOpen(false)}>
                        <Button className="w-full">
                          Login
                        </Button>
                      </Link>
                    )}
                  </div>
                </div>
              </SheetContent>
            </Sheet>
            
            {/* Desktop Buttons */}
            <div className="hidden md:flex items-center gap-2">
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
      </div>
    </nav>
  );
}