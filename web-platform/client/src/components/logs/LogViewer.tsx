import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Terminal, Circle } from "lucide-react";
import { useState, useEffect, useRef } from "react";

interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  module?: string;
  line?: number;
  thread?: string;
}

interface LogViewerProps {
  logs: LogEntry[];
  logLevel: string;
  onLogLevelChange: (level: string) => void;
  isConnected: boolean;
}

const LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];

const getLevelColor = (level: string) => {
  switch (level) {
    case "DEBUG": return "bg-gray-500";
    case "INFO": return "bg-blue-500";
    case "WARNING": return "bg-yellow-500";
    case "ERROR": return "bg-orange-500";
    case "CRITICAL": return "bg-red-500";
    default: return "bg-gray-500";
  }
};

const getLevelBadgeVariant = (level: string): "default" | "secondary" | "destructive" | "outline" => {
  switch (level) {
    case "ERROR":
    case "CRITICAL":
      return "destructive";
    case "WARNING":
      return "outline";
    default:
      return "secondary";
  }
};

export function LogViewer({ logs, logLevel, onLogLevelChange, isConnected }: LogViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Terminal className="h-5 w-5" />
            Device Logs
            {isConnected && <Circle className="h-2 w-2 fill-green-500 text-green-500 animate-pulse" />}
          </CardTitle>
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground">Level:</label>
            <Select value={logLevel} onValueChange={onLogLevelChange}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {LOG_LEVELS.map((level) => (
                  <SelectItem key={level} value={level}>
                    {level}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] w-full rounded-md border bg-black/5 dark:bg-black/20">
          <div ref={scrollRef} className="p-4 font-mono text-xs space-y-1">
            {logs.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                <Terminal className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No logs yet</p>
                <p className="text-xs mt-1">
                  {isConnected 
                    ? `Streaming logs at ${logLevel} level...`
                    : 'Device is offline'}
                </p>
              </div>
            ) : (
              logs.map((log, index) => (
                <div 
                  key={index} 
                  className="flex items-start gap-2 hover:bg-muted/50 px-2 py-1 rounded group"
                >
                  <span className="text-muted-foreground shrink-0 text-[10px]">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <Badge 
                    variant={getLevelBadgeVariant(log.level)}
                    className={`shrink-0 text-[10px] h-5 ${getLevelColor(log.level)} text-white`}
                  >
                    {log.level}
                  </Badge>
                  <span className="text-muted-foreground shrink-0 text-[10px]">
                    {log.logger}
                  </span>
                  <span className="flex-1 break-all">
                    {log.message}
                  </span>
                  {log.module && (
                    <span className="text-muted-foreground shrink-0 text-[10px] opacity-0 group-hover:opacity-100 transition-opacity">
                      {log.module}:{log.line}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        </ScrollArea>
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>{logs.length} log entries</span>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded"
            />
            Auto-scroll
          </label>
        </div>
      </CardContent>
    </Card>
  );
}
