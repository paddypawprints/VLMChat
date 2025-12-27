import { useState, useEffect } from 'react';
import { auth, getSessionId, setSessionId } from '@/lib/api';

interface User {
  id: string;
  email: string;
  name: string;
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user has existing session
    const sessionId = getSessionId();
    if (sessionId) {
      // In a real app, you'd verify the session with the server
      // For now, we'll assume it's valid and restore from localStorage
      const userData = localStorage.getItem('ir-user');
      if (userData) {
        try {
          setUser(JSON.parse(userData));
        } catch (e) {
          console.error('Failed to parse user data:', e);
          setSessionId(null);
        }
      }
    }
    setLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    const response = await auth.login(email, password);
    setUser(response.user);
    localStorage.setItem('ir-user', JSON.stringify(response.user));
    return response;
  };

  const oidcLogin = async (provider: string) => {
    const response = await auth.oidcLogin(provider);
    setUser(response.user);
    localStorage.setItem('ir-user', JSON.stringify(response.user));
    return response;
  };

  const logout = async () => {
    try {
      await auth.logout();
    } catch (e) {
      console.error('Logout error:', e);
    }
    setUser(null);
    localStorage.removeItem('ir-user');
    setSessionId(null);
  };

  return {
    user,
    loading,
    login,
    oidcLogin,
    logout,
    isAuthenticated: !!user,
  };
}