# Design Guidelines for Independent Research

## Design Approach
**System-Based Approach**: Material Design with custom adaptations
- Rationale: Information-dense application with technical users requiring clear data presentation, device connectivity interfaces, and debug modes
- Emphasizes functionality and usability over pure aesthetics
- Modern tech aesthetic appropriate for AI/ML company

## Core Design Elements

### A. Color Palette
**Primary Colors:**
- Dark Mode Primary: 220 85% 65% (Electric blue - representing cutting-edge AI)
- Light Mode Primary: 220 75% 45% 
- Surface Dark: 220 15% 12%
- Surface Light: 220 10% 96%

**Supporting Colors:**
- Success (device connected): 142 70% 45%
- Warning (debug mode): 38 85% 55%
- Error (connection failed): 0 75% 55%
- Text Primary Dark: 220 10% 95%
- Text Primary Light: 220 15% 15%

### B. Typography
**Font Stack:** Inter (Google Fonts)
- Headers: Inter 600 (Semi-bold)
- Body: Inter 400 (Regular)  
- Code/Debug: JetBrains Mono 400
- Button text: Inter 500 (Medium)

### C. Layout System
**Tailwind Spacing Primitives:** 2, 4, 6, 8, 12, 16
- Consistent spacing reduces cognitive load
- p-4, m-6, gap-8 for standard layouts
- p-2 for tight spaces, p-12 for generous sections

### D. Component Library

**Navigation:**
- Top navigation bar with Independent Research logo
- Route indicators for Home, Login, Chat
- Device connection status indicator

**Home/Marketing Page:**
- Hero section with gradient background (220 85% 65% to 220 45% 35%)
- 3 core sections: Hero, Features, Technology showcase
- CTA buttons with primary brand colors
- Feature cards with subtle elevation (shadow-lg)

**Authentication:**
- Clean login form with OpenID Connect styling
- Minimal design focusing on security trust indicators
- Consistent with overall app aesthetic

**Chat Interface:**
- Message bubbles: User (primary color), Assistant (neutral gray)
- Input area with text field, image upload, camera button
- Debug panel (collapsible) with monospace font
- Device connection status always visible

**Edge Device Connection:**
- Device discovery list with connection states
- Visual indicators for connection status
- Technical specifications display area

### E. Interactive Elements
- Buttons: Rounded corners (rounded-md), solid primary for main actions
- Form inputs: Clean borders, focus states with primary color
- Cards: Subtle shadows, hover states with slight elevation increase
- No complex animations - simple transitions only (transition-colors duration-200)

## Special Considerations
- **Debug Mode**: Toggle switch prominently placed, changes interface to show technical data
- **Camera Integration**: Native camera access UI with preview window
- **Device Status**: Always-visible connection indicator in header
- **Responsive**: Mobile-first approach with touch-friendly targets (min 44px)

## Images
- **Hero Image**: Large background image showing edge devices (Raspberry Pi, Jetson) in clean tech environment
- **Feature Icons**: Simple line icons representing AI, edge computing, connectivity
- **Device Images**: Product shots of supported edge devices for connection page

The design emphasizes trust, technical competence, and ease of use while maintaining a modern, professional appearance suitable for an AI technology company.