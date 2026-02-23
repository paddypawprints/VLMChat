import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface DataPoint {
  timestamp: string;
  alive: number;
  cleaned: number;
  leaked: number;
}

interface MemoryLeakChartProps {
  data: DataPoint[];
  title?: string;
}

export function MemoryLeakChart({ data, title = "Memory Tracking" }: MemoryLeakChartProps) {
  // Format data for recharts - show last 20 points
  const chartData = data.slice(-20).map((point, index) => ({
    index,
    time: new Date(point.timestamp).toLocaleTimeString(),
    alive: point.alive,
    leaked: point.leaked,
  }));

  const currentAlive = data.length > 0 ? data[data.length - 1].alive : 0;
  const currentLeaked = data.length > 0 ? data[data.length - 1].leaked : 0;
  const totalCleaned = data.length > 0 ? data[data.length - 1].cleaned : 0;
  
  const hasLeaks = currentLeaked > 0;

  return (
    <Card className="md:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <AlertTriangle className={`h-4 w-4 ${hasLeaks ? 'text-yellow-500' : 'text-muted-foreground'}`} />
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
                label={{ value: 'Objects', angle: -90, position: 'insideLeft', fontSize: 12 }}
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
                dataKey="alive" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                name="Alive"
              />
              {hasLeaks && (
                <Line 
                  type="monotone" 
                  dataKey="leaked" 
                  stroke="hsl(var(--destructive))" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  name="Leaked"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Alive</p>
            <p className="text-lg font-semibold">{currentAlive}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Cleaned</p>
            <p className="text-lg font-semibold text-green-600">{totalCleaned}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Leaked</p>
            <p className={`text-lg font-semibold ${hasLeaks ? 'text-destructive' : 'text-muted-foreground'}`}>
              {currentLeaked}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
