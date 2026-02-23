import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Clock } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface DataPoint {
  timestamp: string;
  value: number;
}

interface DurationChartProps {
  data: DataPoint[];
  title?: string;
}

export function DurationChart({ data, title = "Pipeline Duration" }: DurationChartProps) {
  // Format data for recharts - show last 20 points
  const chartData = data.slice(-20).map((point, index) => ({
    index,
    time: new Date(point.timestamp).toLocaleTimeString(),
    duration: point.value,
  }));

  const avgDuration = data.length > 0 
    ? data.reduce((sum, p) => sum + p.value, 0) / data.length 
    : 0;

  const maxDuration = data.length > 0 
    ? Math.max(...data.map(p => p.value)) 
    : 0;

  return (
    <Card className="md:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Clock className="h-4 w-4" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="time" 
                tick={{ fontSize: 12 }}
                className="text-muted-foreground"
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                className="text-muted-foreground"
                label={{ value: 'ms', angle: -90, position: 'insideLeft', fontSize: 12 }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
              />
              <Line 
                type="monotone" 
                dataKey="duration" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Average</p>
            <p className="text-lg font-semibold">{avgDuration.toFixed(1)} ms</p>
          </div>
          <div>
            <p className="text-muted-foreground">Peak</p>
            <p className="text-lg font-semibold">{maxDuration.toFixed(1)} ms</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
