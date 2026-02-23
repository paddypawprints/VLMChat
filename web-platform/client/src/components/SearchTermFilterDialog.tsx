import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";

interface SearchTermFilter {
  id: string;
  search_string: string;
  category_mask: boolean[];
  category_colors: (string | null)[];
  attribute_mask: boolean[];
  attribute_colors: (string | null)[];
  groq_response?: any;
}

interface SearchTermFilterDialogProps {
  filter: SearchTermFilter | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// COCO categories (80 total)
const COCO_CATEGORIES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

// PA100K attributes (26 total) - must match server order
const PA100K_ATTRIBUTES = [
  "Female", "AgeOver60", "Age18-60", "AgeLess18", "Front", "Side", "Back",
  "Hat", "Glasses", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
  "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
  "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots"
];

export function SearchTermFilterDialog({
  filter,
  open,
  onOpenChange,
}: SearchTermFilterDialogProps) {
  if (!filter) return null;

  const enabledCategories = filter.category_mask
    .map((enabled, idx) => (enabled ? { name: COCO_CATEGORIES[idx], color: filter.category_colors[idx] } : null))
    .filter((cat): cat is { name: string; color: string | null } => cat !== null);

  const enabledAttributes = filter.attribute_mask
    .map((enabled, idx) => (enabled ? { name: PA100K_ATTRIBUTES[idx], color: filter.attribute_colors[idx] } : null))
    .filter((attr): attr is { name: string; color: string | null } => attr !== null);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Filter Details</DialogTitle>
        </DialogHeader>

        <ScrollArea className="max-h-[60vh] pr-4">
          <div className="space-y-4">
            {/* Search String */}
            <div>
              <h4 className="text-sm font-semibold mb-2">Search Query</h4>
              <p className="text-sm text-muted-foreground italic">"{filter.search_string}"</p>
            </div>

            <Separator />

            <div className="space-y-6">
              {/* Categories */}
              <div>
                <h4 className="text-sm font-semibold mb-3">
                  YOLO Categories ({enabledCategories.length}/{COCO_CATEGORIES.length})
                </h4>
                {enabledCategories.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {enabledCategories.map((cat) => (
                      <Badge key={cat.name} variant="secondary" className="gap-1">
                        {cat.name}
                        {cat.color && (
                          <span
                            className="w-3 h-3 rounded-full border"
                            style={{ backgroundColor: cat.color }}
                            title={cat.color}
                          />
                        )}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No categories selected</p>
                )}
              </div>

              <Separator />

              {/* Attributes */}
              <div>
                <h4 className="text-sm font-semibold mb-3">
                  Person Attributes ({enabledAttributes.length}/{PA100K_ATTRIBUTES.length})
                </h4>
                {enabledAttributes.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {enabledAttributes.map((attr) => (
                      <Badge key={attr.name} variant="outline" className="gap-1">
                        {attr.name}
                        {attr.color && (
                          <span
                            className="w-3 h-3 rounded-full border"
                            style={{ backgroundColor: attr.color }}
                            title={attr.color}
                          />
                        )}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No attributes selected</p>
                )}
              </div>

              <Separator />

              {/* Technical Details */}
              <div>
                <h4 className="text-sm font-semibold mb-3">Technical Details</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Category Vector:</span>
                    <p className="font-mono text-xs mt-1 truncate">
                      {filter.category_mask.filter(Boolean).length} bits set
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Attribute Vector:</span>
                    <p className="font-mono text-xs mt-1 truncate">
                      {filter.attribute_mask.filter(Boolean).length} bits set
                    </p>
                  </div>
                </div>
              </div>

              {/* Groq Response */}
              {filter.groq_response && (
                <>
                  <Separator />
                  <div>
                    <h4 className="text-sm font-semibold mb-3">Groq LLM Response</h4>
                    <pre className="text-xs bg-muted p-3 rounded-md overflow-x-auto">
                      {JSON.stringify(filter.groq_response, null, 2)}
                    </pre>
                  </div>
                </>
              )}
            </div>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
