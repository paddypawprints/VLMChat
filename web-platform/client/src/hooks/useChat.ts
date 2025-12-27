import { useState, useEffect } from 'react';
import { chat } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  images?: string[];
  debug?: {
    systemPrompt?: string;
    modelInputs?: Record<string, any>;
    modelOutputs?: Record<string, any>;
    processingTime?: number;
    device?: Record<string, any>;
  };
  createdAt: Date;
}

export function useChat(deviceId?: string | null, isAuthenticated = false) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const { toast } = useToast();

  const fetchMessages = async () => {
    if (!isAuthenticated || !deviceId) return;
    
    try {
      setLoading(true);
      const data = await chat.getMessages(deviceId);
      setMessages(data.map((msg: any) => ({
        ...msg,
        createdAt: new Date(msg.createdAt)
      })));
    } catch (error) {
      console.error('Failed to fetch messages:', error);
      toast({
        title: "Error",
        description: "Failed to load chat history",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (message: string, images: File[] = [], debugMode = false) => {
    if (!isAuthenticated || !deviceId || (!message.trim() && images.length === 0)) return;

    try {
      setSending(true);
      const response = await chat.sendMessage(message, images, deviceId, debugMode);
      
      // Add both user and AI messages to state
      const newMessages = [response.userMessage, response.aiMessage].map((msg: any) => ({
        ...msg,
        createdAt: new Date(msg.createdAt)
      }));
      
      setMessages(prev => [...prev, ...newMessages]);
    } catch (error) {
      console.error('Failed to send message:', error);
      toast({
        title: "Error",
        description: "Failed to send message",
        variant: "destructive",
      });
    } finally {
      setSending(false);
    }
  };

  useEffect(() => {
    fetchMessages();
  }, [isAuthenticated, deviceId]);

  return {
    messages,
    loading,
    sending,
    sendMessage,
    refreshMessages: fetchMessages,
  };
}