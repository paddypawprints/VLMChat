import { HeroSection } from '../HeroSection';
import { ThemeProvider } from '../ThemeProvider';

export default function HeroSectionExample() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-background">
        <HeroSection />
      </div>
    </ThemeProvider>
  );
}