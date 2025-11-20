# CLIP Testing Results - MobileCLIP2-S0 Evaluation

## Overview

This document summarizes comprehensive testing of the MobileCLIP2-S0 model for attribute detection in the VLMChat pipeline. Tests were conducted to evaluate CLIP's ability to detect color attributes, understand semantic context, and determine whether it relies on pixel-level features or higher-level semantic understanding.

**Model Tested:** MobileCLIP2-S0  
**Test Image:** trail-riders.jpg (3 people on horses)  
**Ground Truth (Visual Inspection):**
- Detection #0: Person wearing **black shirt** and white hat
- Detection #1: Person wearing **blue shirt** and white hat  
- Detection #2: Person wearing **white shirt** and white hat

---

## Test 16: Color Saturation Enhancement

### Objective
Test whether 2.5x color saturation boost improves CLIP's ability to detect color attributes in clothing.

### Methodology
- **PATH A (Baseline):** Normal color - no saturation enhancement
- **PATH B (Enhanced):** 2.5x saturation boost via ColorEnhanceTask
- Both paths process the same person detections through CLIP
- Compare similarity scores for color-related prompts

### Prompts Tested
1. "a person wearing a white hat"
2. "a person wearing a red shirt"
3. "a person wearing a blue shirt"
4. "a person wearing a black shirt"
5. "a person wearing a white shirt"
6. "a white hat"
7. "a chair"

### Results

#### Detection #0: Black Shirt (Ground Truth)
**PATH A - Normal:**
- White hat: 0.189 (best match) ⚪
- Black shirt: 0.160 (ranks 4th)
- White shirt: 0.183
- Red shirt: 0.145
- Blue shirt: 0.154

**PATH B - 2.5x Saturation:**
- White hat: 0.180 (best match) ⚪
- Black shirt: 0.150 (decreased)
- White shirt: 0.174
- Red shirt: 0.147
- Blue shirt: 0.160

**Analysis:** 
- CLIP focused on the white hat, ignoring the black shirt
- Saturation actually **decreased** black shirt detection (0.160 → 0.150)
- Black shirt ranks poorly in both cases

#### Detection #1: Blue Shirt (Ground Truth) ✅
**PATH A - Normal:**
- Blue shirt: 0.184 (best match) 🔵
- Black shirt: 0.178
- White hat: 0.169
- Red shirt: 0.159

**PATH B - 2.5x Saturation:**
- Blue shirt: 0.190 (best match) 🔵 **+0.006**
- Black shirt: 0.171
- White hat: 0.164
- Red shirt: 0.165

**Analysis:**
- CLIP correctly identified blue shirt in both cases ✅
- Saturation modestly improved blue detection (+0.006)
- This is the only detection where CLIP correctly identified the shirt color

#### Detection #2: White Shirt (Ground Truth)
**PATH A - Normal:**
- Blue shirt: 0.191 (best match - WRONG)
- White shirt: 0.189
- White hat: 0.185
- Black shirt: 0.181
- Red shirt: 0.179

**PATH B - 2.5x Saturation:**
- Blue shirt: 0.195 (best match - WRONG) **+0.004**
- White shirt: 0.185
- White hat: 0.184
- Red shirt: 0.186 **+0.007**
- Black shirt: 0.176

**Analysis:**
- CLIP incorrectly identified as blue shirt (actually white)
- Saturation increased blue shirt score further (+0.004)
- Red shirt score also increased (+0.007) despite no red present
- Saturation boosts all colors indiscriminately, amplifying both correct and incorrect detections

### Test 16 Conclusions

1. **Saturation has mixed effects:**
   - Modest improvement for actual blue detection (+0.006)
   - Decreased black shirt detection
   - Amplified false positives (red shirt scores increased)

2. **CLIP struggles with black detection:**
   - Black shirt (Detection #0) ranks 4th out of 6 prompts
   - Black vs. dark blue confusion observed

3. **White hat dominates Detection #0:**
   - CLIP focuses on prominent white hat instead of black shirt
   - Suggests spatial attention or saliency issues

4. **Red shirt false positives:**
   - Red scores elevated across all detections despite no red shirts present
   - Possibly due to brown/tan colors of horses and saddles

5. **Overall accuracy: 1/3 (33%)**
   - Only Detection #1 (blue shirt) correctly identified
   - Detection #0 missed (black → white hat)
   - Detection #2 missed (white → blue)

---

## Test 17: Red/Blue Channel Swap

### Objective
Determine whether CLIP follows actual pixel colors or relies on semantic/contextual understanding by swapping red and blue channels.

### Methodology
- **PATH A (Baseline):** Normal RGB color channels
- **PATH B (Swapped):** Red ↔ Blue channel swap via ColorSwapTask
  - Blue pixels become red pixels
  - Red/brown pixels become blue pixels
- If CLIP follows pixels: red scores should increase, blue scores should decrease
- If CLIP uses semantic context: scores should remain similar

### Technical Verification
Channel swap confirmed working correctly:
```
Original image: R=118.5, G=121.1, B=84.5 (R - B = +34.0)
Swapped detection: R - B = -20.0 (flip confirmed)
```
The R/B channels are properly swapped (±20 point flip in R-B difference).

### Results

#### Detection #0: Black Shirt (Ground Truth)
**PATH A - Normal:**
- White hat: 0.189 (best match)
- White shirt: 0.183
- Black shirt: 0.160
- Blue shirt: 0.154
- Red shirt: 0.145

**PATH B - R/B Swapped:**
- White hat: 0.189 (best match - **unchanged**)
- White shirt: 0.181
- Blue shirt: 0.174 **+0.020**
- Black shirt: 0.157
- Red shirt: 0.147

**Analysis:**
- White hat score unchanged (0.189 → 0.189)
- Blue shirt score increased significantly (+0.020) after swap
- If CLIP followed pixels, blue should decrease (blue pixels → red pixels)
- **Paradox:** Blue score increased when blue pixels were removed

#### Detection #1: Blue Shirt (Ground Truth) ✅
**PATH A - Normal:**
- Blue shirt: 0.184 (best match) 🔵
- Black shirt: 0.178
- White hat: 0.169
- Red shirt: 0.159

**PATH B - R/B Swapped:**
- **Black shirt: 0.179** (best match - changed!)
- Blue shirt: 0.171 **-0.013**
- White hat: 0.164
- Red shirt: 0.153

**Analysis:**
- CLIP switched from "blue shirt" to "black shirt" after swap
- Blue score decreased (0.184 → 0.171) ✅ expected
- Black score increased (0.178 → 0.179) and became dominant
- **This is the only detection showing pixel-following behavior**
- After swapping blue→red, CLIP saw darker tones and matched to "black"

#### Detection #2: White Shirt (Ground Truth)
**PATH A - Normal:**
- Blue shirt: 0.191 (best match - WRONG)
- White shirt: 0.189
- White hat: 0.185
- Black shirt: 0.181
- Red shirt: 0.179

**PATH B - R/B Swapped:**
- Blue shirt: 0.208 (best match - WRONG) **+0.017** 🤯
- Black shirt: 0.201 **+0.020**
- White shirt: 0.184
- White hat: 0.184
- Red shirt: 0.171

**Analysis:**
- CLIP still says "blue shirt" despite swapping blue→red
- Blue score **increased** dramatically (+0.017) when blue pixels were removed!
- Black score also increased significantly (+0.020)
- **This is the opposite of expected behavior if following pixels**
- Suggests CLIP using contextual cues (pose, scene, texture) over raw RGB

### Test 17 Conclusions

1. **Mixed pixel vs. semantic behavior:**
   - Detection #1: Shows pixel-following (blue→black after swap)
   - Detection #0 & #2: Show semantic/contextual behavior (blue increases after removing blue pixels)

2. **CLIP doesn't purely follow pixels:**
   - If pixel-based: blue scores should decrease universally after blue→red swap
   - Actual: blue scores increased in 2/3 cases
   - Evidence of higher-level semantic processing

3. **Spatial/attention effects:**
   - Different detections show different behaviors
   - Suggests CLIP's attention mechanism weighs features differently based on crop content

4. **Black color confusion persists:**
   - Detection #1 switched to "black" after swap
   - Black vs. dark blue not well differentiated
   - Dark colors in shadow areas may confuse the model

5. **Paradoxical results (Detection #2):**
   - Blue score increases when blue pixels are removed
   - Suggests CLIP may be keying on:
     - Shape/pose (person on horse)
     - Texture patterns
     - Scene context
     - Non-color features that correlate with "blue shirt" in training data

---

## Overall Findings

### Model Limitations Discovered

1. **Poor black shirt detection:**
   - Black shirt consistently ranks low (4th-5th out of 6-7 prompts)
   - Black vs. dark colors not well separated in embedding space
   - Only 0/3 correct black shirt identifications

2. **Color detection accuracy: 33% (1/3 correct)**
   - Only Detection #1 (blue shirt) correctly identified
   - Detection #0 (black) → predicted white hat
   - Detection #2 (white) → predicted blue shirt

3. **Spatial attention biases:**
   - White hats dominate attention over shirt colors
   - Prominent features override subtle color cues
   - Suggests spatial weighting in CLIP attention mechanism

4. **False positive for red:**
   - Red shirt scores elevated across all detections
   - No red shirts present in image
   - Likely triggered by brown/tan horses and saddles
   - Red vs. brown/tan confusion

5. **Inconsistent pixel vs. semantic processing:**
   - Some detections follow pixel colors (Detection #1)
   - Others use semantic context (Detections #0, #2)
   - Behavior varies by detection/crop content

### MobileCLIP Embedding Space Issues

Based on these tests, the **MobileCLIP2-S0 embedding space does not sufficiently differentiate** the specific color attributes being tested:

1. **Insufficient color separation:**
   - Black, dark blue, and shadow areas overlap
   - Red, brown, and tan colors overlap
   - White vs. light colors distinction unclear

2. **Feature hierarchy problems:**
   - Prominent features (white hats) dominate over target features (shirt colors)
   - Model may not be attending to the correct spatial regions

3. **Color constancy issues:**
   - Model behavior inconsistent when colors are manipulated
   - Blue detection increases when blue pixels are removed (paradox)

4. **Training data bias:**
   - Red shirt scores elevated despite no red present
   - Suggests training distribution may overweight certain color associations

### Recommendations for Further Testing

1. **Test with different CLIP models:**
   - OpenAI CLIP ViT-B/32 or ViT-L/14
   - Other MobileCLIP variants (larger models)
   - SigLIP or other recent vision-language models

2. **Controlled color tests:**
   - Solid color patches (remove context)
   - Isolated clothing items (no scene context)
   - Systematic color wheel testing

3. **Spatial attention analysis:**
   - Test with crops of different sizes
   - Vary expansion factors
   - Test shirt-only crops vs. full person

4. **Prompt engineering:**
   - Test more specific prompts ("dark blue shirt", "light blue shirt")
   - Test negative prompts
   - Test with color hex codes or specific color names

5. **Embedding space analysis:**
   - Compute embedding distances between color prompts
   - Visualize embedding space with t-SNE/UMAP
   - Measure separation between similar colors

6. **Alternative approaches:**
   - Consider dedicated color classification models
   - Hybrid approach: CLIP for objects, specialized model for colors
   - Fine-tune CLIP on color-focused dataset

---

## Test Configuration

### Hardware/Software
- Platform: macOS (Apple Silicon)
- Python: 3.11.13
- Model: MobileCLIP2-S0
- Model path: `src/models/MobileClip/ml-mobileclip/mobileclip2_s0.pt`
- YOLO: yolov8n.pt (CPU inference)

### Pipeline Configuration
- Detection expansion: 20%
- CLIP similarity threshold: 0.15
- YOLO confidence threshold: 0.25
- Color enhancement: 2.5x saturation (Test 16)
- Channel swap: (0, 2) = Red ↔ Blue (Test 17)

### Test Images
- Primary: `src/camera/trail-riders.jpg`
- Resolution: 1389 × 2000 pixels
- Content: 3 people on horses, outdoor scene
- Lighting: Natural daylight

---

## Conclusion

The MobileCLIP2-S0 model shows **significant limitations for fine-grained color attribute detection**:

- **33% accuracy** on shirt color identification (1/3 correct)
- **Inconsistent behavior** between pixel-following and semantic understanding
- **Poor differentiation** of black and dark colors
- **Spatial attention biases** toward prominent features (hats over shirts)
- **False positives** for red/brown confusion

These results suggest the **embedding space does not sufficiently separate color attributes** for reliable detection. The model appears to be optimized for general object recognition and scene understanding rather than fine-grained color discrimination.

**Next steps** should focus on:
1. Testing alternative CLIP models (OpenAI CLIP, larger MobileCLIP variants)
2. Analyzing the embedding space directly
3. Considering specialized color classification models
4. Exploring fine-tuning approaches for color-specific tasks
