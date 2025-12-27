import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Shield, Github, Chrome, Apple } from "lucide-react";

interface LoginFormProps {
  onLogin?: (email: string, password: string) => void;
  onOIDCLogin?: (provider: string) => void;
  loading?: boolean;
  error?: string;
}

export function LoginForm({ onLogin, onOIDCLogin, loading = false, error }: LoginFormProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onLogin) {
      onLogin(email, password);
    }
  };

  const providers = [
    { name: "Google", icon: Chrome, id: "google" },
    { name: "GitHub", icon: Github, id: "github" },
    { name: "Apple", icon: Apple, id: "apple" }
  ];

  return (
    <div className="min-h-screen flex items-center justify-center bg-muted/30 p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className="p-3 rounded-full bg-primary/10">
              <Shield className="h-6 w-6 text-primary" />
            </div>
          </div>
          <CardTitle className="text-2xl">Welcome Back</CardTitle>
          <CardDescription>
            Sign in to access your edge AI dashboard
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* OpenID Connect Providers */}
          <div className="space-y-3">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">Continue with</span>
              </div>
            </div>
            
            {providers.map((provider) => (
              <Button
                key={provider.id}
                variant="outline"
                className="w-full gap-2"
                onClick={() => onOIDCLogin?.(provider.id)}
                disabled={loading}
                data-testid={`button-login-${provider.id}`}
              >
                <provider.icon className="h-4 w-4" />
                {provider.name}
              </Button>
            ))}
          </div>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-card px-2 text-muted-foreground">Or continue with email</span>
            </div>
          </div>

          {/* Email/Password Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="your@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                data-testid="input-email"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                data-testid="input-password"
              />
            </div>
            
            <Button 
              type="submit" 
              className="w-full" 
              disabled={loading}
              data-testid="button-submit-login"
            >
              {loading ? "Signing in..." : "Sign In"}
            </Button>
          </form>

          <div className="text-center">
            <Badge variant="secondary" className="text-xs">
              Demo Mode - Use any credentials
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}