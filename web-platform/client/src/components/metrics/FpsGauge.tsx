import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity } from "lucide-react";

interface FpsGaugeProps {
  value: number;
  max?: number;
}

export function FpsGauge({ value, max = 60 }: FpsGaugeProps) {
  const percentage = Math.min((value / max) * 100, 100);
  const strokeDasharray = 2 * Math.PI * 45; // circumference for radius 45
  const strokeDashoffset = strokeDasharray * (1 - percentage / 100);
  
  // Color based on FPS performance
  const getColor = (fps: number) => {
    if (fps >= 25) return "text-green-500";
    if (fps >= 15) return "text-yellow-500";
    return "text-red-500";
  };

  const color = getColor(value);
  
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4" />
          Frames Per Second
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative flex items-center justify-center">
          <svg width="120" height="120" className="-rotate-90">
            {/* Background circle */}
            <circle
              cx="60"
              cy="60"
              r="45"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              className="text-muted opacity-20"
            />
            {/* Progress circle */}
            <circle
              cx="60"
              cy="60"
              r="45"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              strokeLinecap="round"
              className={`${color} transition-all duration-500`}
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-3xl font-bold ${color}`}>
              {value.toFixed(1)}
            </span>
            <span className="text-xs text-muted-foreground">FPS</span>
          </div>
        </div>
        <div className="mt-4 flex justify-between text-xs text-muted-foreground">
          <span>0</span>
          <span>{max}</span>
        </div>
      </CardContent>
    </Card>
  );
}
