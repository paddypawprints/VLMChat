/**
 * Canvas component for displaying snapshots with detection bounding boxes
 */
import { useEffect, useRef } from 'react';

interface Detection {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  category: string;
  category_id?: number;
  children?: Detection[];
}

interface SnapshotCanvasProps {
  imageData: string; // Base64 encoded image
  detections?: Detection[];
}

// Color palette for different categories
const CATEGORY_COLORS: Record<string, string> = {
  person: '#FF6B6B',
  car: '#4ECDC4',
  truck: '#45B7D1',
  dog: '#FFA07A',
  cat: '#98D8C8',
  bird: '#F7DC6F',
  bicycle: '#BB8FCE',
  motorcycle: '#85C1E9',
};

const DEFAULT_COLOR = '#00FF00';

export function SnapshotCanvas({ imageData, detections = [] }: SnapshotCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    console.log('[SnapshotCanvas] Rendering with', detections.length, 'detections');
    if (!canvasRef.current || !imageData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Load image
    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw detections recursively (handles tree structure)
      const drawDetection = (det: Detection) => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Get color for category
        const color = CATEGORY_COLORS[det.category] || DEFAULT_COLOR;

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        const label = `${det.category} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = '14px Inter, sans-serif';
        const textMetrics = ctx.measureText(label);
        const textHeight = 20;
        const padding = 4;

        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - textHeight - padding, textMetrics.width + padding * 2, textHeight + padding);

        // Draw label text
        ctx.fillStyle = '#000000';
        ctx.fillText(label, x1 + padding, y1 - padding - 2);

        // Draw children (clustered detections)
        if (det.children && det.children.length > 0) {
          det.children.forEach(child => drawDetection(child));
        }
      };

      // Draw all detections
      detections.forEach(det => drawDetection(det));
    };

    img.src = `data:image/jpeg;base64,${imageData}`;
  }, [imageData, detections]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full object-contain"
      style={{ maxWidth: '100%', maxHeight: '100%' }}
    />
  );
}
