import { Badge } from "@/components/ui/badge";
import { CheckCircle2, AlertTriangle, Clock, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface VlmStatusBadgeProps {
  vlmRequired?: boolean;
  vlmStatus?: string | null; // 'confirmed' | 'timeout' | 'invalid_response' | null
  vlmResponse?: string | null;
  className?: string;
}

export function VlmStatusBadge({
  vlmRequired,
  vlmStatus,
  vlmResponse,
  className,
}: VlmStatusBadgeProps) {
  // 4-Color Status System:
  // GREEN: vlmStatus === 'confirmed' (VLM said yes)
  // BLUE: vlmRequired === false (VLM not required)
  // YELLOW: vlmStatus === 'timeout' (VLM busy/queue full)
  // RED: vlmStatus === 'invalid_response' (VLM gave non-yes/no answer)
  
  const getStatusInfo = () => {
    // GREEN: VLM confirmed
    if (vlmStatus === 'confirmed') {
      return {
        icon: CheckCircle2,
        text: "VLM Confirmed",
        variant: "success" as const,
        className: "bg-green-100 text-green-800 border-green-300 dark:bg-green-900 dark:text-green-100 dark:border-green-700",
      };
    }
    
    // YELLOW: VLM timeout
    if (vlmStatus === 'timeout') {
      return {
        icon: Clock,
        text: "VLM Timeout",
        variant: "warning" as const,
        className: "bg-yellow-100 text-yellow-800 border-yellow-300 dark:bg-yellow-900 dark:text-yellow-100 dark:border-yellow-700",
      };
    }
    
    // RED: VLM invalid response
    if (vlmStatus === 'invalid_response') {
      return {
        icon: XCircle,
        text: "VLM Invalid",
        variant: "destructive" as const,
        className: "bg-red-100 text-red-800 border-red-300 dark:bg-red-900 dark:text-red-100 dark:border-red-700",
        tooltip: vlmResponse ? `Response: ${vlmResponse}` : undefined,
      };
    }
    
    // BLUE: VLM not required (direct detection)
    if (!vlmRequired && !vlmStatus) {
      return {
        icon: CheckCircle2,
        text: "Detected",
        variant: "default" as const,
        className: "bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900 dark:text-blue-100 dark:border-blue-700",
      };
    }
    
    // Default fallback
    return null;
  };

  const statusInfo = getStatusInfo();
  
  if (!statusInfo) {
    return null;
  }

  const Icon = statusInfo.icon;

  return (
    <Badge className={cn(statusInfo.className, "gap-1", className)}>
      <Icon className="h-3 w-3" />
      {statusInfo.text}
    </Badge>
  );
}
