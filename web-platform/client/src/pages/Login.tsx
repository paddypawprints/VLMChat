import { useState } from "react";
import { useLocation } from "wouter";
import { LoginForm } from "@/components/LoginForm";

interface LoginProps {
  onLogin?: (email: string, password: string) => Promise<any>;
  onOIDCLogin?: (provider: string) => Promise<any>;
}

export default function Login({ onLogin, onOIDCLogin }: LoginProps) {
  const [, setLocation] = useLocation();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (email: string, password: string) => {
    try {
      setLoading(true);
      setError("");
      
      await onLogin?.(email, password);
      setLocation('/devices');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleOIDCLogin = async (provider: string) => {
    try {
      setLoading(true);
      setError("");
      
      await onOIDCLogin?.(provider);
      setLocation('/devices');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <LoginForm
      onLogin={handleLogin}
      onOIDCLogin={handleOIDCLogin}
      loading={loading}
      error={error}
    />
  );
}