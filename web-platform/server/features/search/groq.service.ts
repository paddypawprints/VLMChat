/**
 * Groq NLP service for parsing natural language queries
 */
import { Groq } from 'groq-sdk';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { secretsManager } from '../../secrets';
import { validateGroqResponse } from '../../validation';
import { fewShotExamples } from '../../groq-few-shot';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// COCO categories in order (80 total)
const COCO_CATEGORIES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
  "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

// PA-100K attributes in order (26 total)
const PA100K_ATTRIBUTES = [
  "Female", "AgeOver60", "Age18-60", "AgeLess18", "Front", "Side", "Back",
  "Hat", "Glasses", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
  "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
  "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots"
];

/**
 * Convert RGB array to hex color
 */
function rgbToHex(rgb: number[]): string {
  return '#' + rgb.map(c => c.toString(16).padStart(2, '0')).join('');
}

/**
 * Extract color name from RGB value (simple heuristic mapping)
 */
function rgbToColorName(rgb: number[]): string {
  const [r, g, b] = rgb;
  
  // Convert to HSV for better color classification
  const rNorm = r / 255;
  const gNorm = g / 255;
  const bNorm = b / 255;
  
  const max = Math.max(rNorm, gNorm, bNorm);
  const min = Math.min(rNorm, gNorm, bNorm);
  const delta = max - min;
  
  // Value (brightness)
  const v = max;
  
  // Saturation
  const s = max === 0 ? 0 : delta / max;
  
  // Hue
  let h = 0;
  if (delta !== 0) {
    if (max === rNorm) {
      h = ((gNorm - bNorm) / delta + (gNorm < bNorm ? 6 : 0)) / 6;
    } else if (max === gNorm) {
      h = ((bNorm - rNorm) / delta + 2) / 6;
    } else {
      h = ((rNorm - gNorm) / delta + 4) / 6;
    }
  }
  h = h * 360; // Convert to degrees
  
  // Low saturation = gray/white/black
  if (s < 0.2) {
    if (v > 0.8) return 'white';
    if (v < 0.3) return 'black';
    return 'gray';
  }
  
  // Map hue to color names
  if (h < 15 || h >= 345) return 'red';
  if (h >= 15 && h < 45) return 'orange';
  if (h >= 45 && h < 75) return 'yellow';
  if (h >= 75 && h < 155) return 'green';
  if (h >= 155 && h < 200) return 'cyan';
  if (h >= 200 && h < 260) return 'blue';
  if (h >= 260 && h < 300) return 'purple';
  if (h >= 300 && h < 345) return 'pink';
  
  return 'unknown';
}

export interface GroqFilterResult {
  categoryMask: boolean[];
  categoryColors: (string | null)[];
  attributeMask: boolean[];
  attributeColors: (string | null)[];
  colorRequirements: Record<number, Record<string, number[][]>>;  // {category_id: {region: [[r,g,b], ...]}}
  groqResponse: any;
}

/**
 * Parse natural language search string using Groq API
 * Returns filter bitmasks and raw Groq response
 */
export async function parseWithGroq(searchString: string): Promise<GroqFilterResult> {
  try {
    // Get Groq API key from secrets manager
    const groqApiKey = await secretsManager.get('GROQ_API_KEY');
    
    // Load system prompt from project root
    const systemPromptPath = join(__dirname, '../../../system_prompt.txt');
    const systemPrompt = await readFile(systemPromptPath, 'utf-8');
    
    // Initialize Groq client
    const groq = new Groq({ apiKey: groqApiKey });
    
    // Call Groq API with few-shot examples
    const chatCompletion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        { role: "system", content: systemPrompt },
        ...fewShotExamples,
        { role: "user", content: searchString }
      ],
      temperature: 0.1,
      max_tokens: 2000,
      response_format: { type: "json_object" }
    });
    
    const responseContent = chatCompletion.choices[0]?.message?.content;
    if (!responseContent) {
      throw new Error('No response from Groq API');
    }
    
    const parsedResponse = JSON.parse(responseContent);
    console.log('[Groq] Parsed response:', JSON.stringify(parsedResponse, null, 2));
    
    // Validate response against schema
    const validation = validateGroqResponse(parsedResponse);
    if (!validation.valid) {
      throw new Error(`Groq response validation failed: ${validation.errors?.join(', ')}`);
    }
    
    // Convert Groq response to filter masks
    const categoryMask = Array(80).fill(false);
    const categoryColors: (string | null)[] = Array(80).fill(null);
    const attributeMask = Array(26).fill(false);
    const attributeColors: (string | null)[] = Array(26).fill(null);
    const colorRequirements: Record<number, Record<string, number[][]>> = {};  // {category_id: {region: [[r,g,b]]}}
    
    // Process first search term (usually only one)
    if (parsedResponse.search_terms && parsedResponse.search_terms.length > 0) {
      const searchTerm = parsedResponse.search_terms[0];
      
      for (const obj of searchTerm.objects) {
        // Find category index
        const categoryIndex = COCO_CATEGORIES.indexOf(obj.category);
        if (categoryIndex !== -1) {
          categoryMask[categoryIndex] = true;
          
          // Extract color requirements from colors field
          if (obj.colors && Object.keys(obj.colors).length > 0) {
            // For person: colors are keyed by region (e.g., "middle-top": [255, 255, 255])
            // Keep RGB arrays for precise HSV tolerance matching
            const regionColors: Record<string, number[][]> = {};
            
            for (const [region, rgb] of Object.entries(obj.colors)) {
              if (!regionColors[region]) {
                regionColors[region] = [];
              }
              regionColors[region].push(rgb);
            }
            
            if (Object.keys(regionColors).length > 0) {
              colorRequirements[categoryIndex] = regionColors;
            }
          }
          
          // Handle category-level color (for non-person objects)
          if (obj.colors && !obj.attributes) {
            // Object-level color (e.g., "red car")
            const colorKeys = Object.keys(obj.colors);
            if (colorKeys.length > 0) {
              const rgb = obj.colors[colorKeys[0]];
              categoryColors[categoryIndex] = rgbToHex(rgb);
            }
          }
        }
        
        // Handle person attributes
        if (obj.category === 'person' && obj.attributes) {
          for (const attr of obj.attributes) {
            const attrIndex = PA100K_ATTRIBUTES.indexOf(attr);
            if (attrIndex !== -1) {
              attributeMask[attrIndex] = true;
              
              // Handle attribute-specific color
              if (obj.colors && obj.colors[attr]) {
                attributeColors[attrIndex] = rgbToHex(obj.colors[attr]);
              }
            }
          }
        }
      }
    }
    
    return {
      categoryMask,
      categoryColors,
      attributeMask,
      attributeColors,
      colorRequirements,
      groqResponse: parsedResponse,
    };
    
  } catch (error) {
    console.error('[Groq] Parse error:', error);
    
    // All errors propagate - we cannot create search terms without valid parsing
    if (error instanceof Error) {
      throw new Error('Failed to parse search string: ' + error.message);
    }
    throw error;
  }
}
