import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Link } from "wouter";
import { Cpu, Zap, Shield, Brain, DollarSign, Wifi, MessageSquare, TrendingDown } from "lucide-react";
import heroImage from "@assets/generated_images/Edge_computing_hero_background_74e759e8.png";
import devicesImage from "@assets/generated_images/Edge_device_product_showcase_f3c3664d.png";

export function HeroSection() {
  const features = [
    {
      icon: Wifi,
      title: "Works Offline",
      description: "When connectivity fails or latency is critical, your AI keeps working locally"
    },
    {
      icon: MessageSquare,
      title: "No-Code AI Behavior",
      description: "Configure vision-language models with simple English—no data scientists or programming required"
    },
    {
      icon: DollarSign,
      title: "Zero Incremental Cost",
      description: "Pay only for your edge device—no expensive foundation model API calls or cloud computing costs"
    },
    {
      icon: TrendingDown,
      title: "Fixed Cost Scaling",
      description: "Make unlimited AI calls for the cost of your device, while cloud costs increase with every request"
    }
  ];

  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="relative min-h-[600px] flex items-center justify-center overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url(${heroImage})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-background/90 via-background/70 to-background/90" />
        </div>
        
        <div className="relative z-10 container mx-auto px-4 text-center">
          <Badge variant="secondary" className="mb-4" data-testid="badge-hero">
            Zero-Cost AI · No Cloud Dependencies
          </Badge>
          
          <h1 className="text-4xl md:text-6xl font-bold mb-6 max-w-4xl mx-auto">
            AI That Works
            <span className="text-primary block mt-2">Without the Cloud</span>
          </h1>
          
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Run powerful vision-language models locally when connectivity fails or latency matters. 
            Set AI behavior with simple English instructions—no data scientists or expensive API calls required.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/login" data-testid="link-get-started">
              <Button size="lg" className="text-lg px-8">
                Get Started
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="text-lg px-8" data-testid="button-learn-more">
              Learn More
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-muted/30">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-4">
            Why Edge AI Changes Everything
          </h2>
          <p className="text-lg text-muted-foreground text-center mb-12 max-w-3xl mx-auto">
            Traditional cloud AI creates dependencies, increases costs with usage, and requires expensive expertise. 
            Edge AI breaks these constraints with local processing that scales at zero incremental cost.
          </p>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {features.map((feature, index) => (
              <Card key={index} className="hover-elevate">
                <CardContent className="p-6 text-center">
                  <feature.icon className="h-12 w-12 text-primary mx-auto mb-4" />
                  <h3 className="font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Showcase */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-6">
                Supported Edge Devices
              </h2>
              <p className="text-lg text-muted-foreground mb-6">
                Our AI solutions are optimized to run efficiently on a variety of edge computing platforms, 
                bringing enterprise-grade language models to the edge.
              </p>
              <ul className="space-y-3">
                <li className="flex items-center gap-3">
                  <Badge variant="outline">✓</Badge>
                  <span>Raspberry Pi 4/5</span>
                </li>
                <li className="flex items-center gap-3">
                  <Badge variant="outline">✓</Badge>
                  <span>NVIDIA Jetson Nano/Xavier</span>
                </li>
                <li className="flex items-center gap-3">
                  <Badge variant="outline">✓</Badge>
                  <span>Intel Neural Compute Stick</span>
                </li>
                <li className="flex items-center gap-3">
                  <Badge variant="outline">✓</Badge>
                  <span>Google Coral Dev Board</span>
                </li>
              </ul>
            </div>
            
            <div className="relative">
              <img 
                src={devicesImage} 
                alt="Edge Computing Devices" 
                className="w-full rounded-lg shadow-lg"
              />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}