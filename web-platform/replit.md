# Independent Research - Edge AI Platform

## Overview

Independent Research is an edge AI platform that enables users to deploy and interact with large language models (LLMs) directly on edge devices like Raspberry Pi and NVIDIA Jetson. The platform provides a web-based interface for device management, AI chat functionality, and real-time monitoring of edge computing resources. Users can connect their edge devices, chat with AI models running locally on those devices, and leverage advanced features like camera integration for multimodal AI interactions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript using Vite as the build tool
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: React hooks with @tanstack/react-query for server state management
- **UI Framework**: Shadcn/ui components built on Radix UI primitives with Tailwind CSS
- **Styling**: Material Design inspired system with custom adaptations for technical users
- **Theme System**: Dark/light mode support with CSS custom properties

### Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **API Design**: RESTful endpoints under `/api` namespace
- **File Uploads**: Multer middleware for handling image uploads (10MB limit)
- **Session Management**: Simple in-memory session storage (development setup)
- **Middleware**: Custom logging, authentication, and error handling

### Data Storage Solutions
- **ORM**: Drizzle ORM for type-safe database operations
- **Database**: PostgreSQL (configured via Neon Database serverless)
- **Schema**: Three main entities - users, devices, and chat messages
- **Migrations**: Drizzle Kit for schema migrations and database management

### Authentication and Authorization
- **Session-based**: Custom session management with session IDs stored in headers
- **Multi-provider Support**: Email/password and OIDC providers (Google, GitHub, Apple)
- **Route Protection**: Middleware-based authentication checks for protected routes
- **Client Storage**: Session persistence via localStorage with automatic session restoration

### Device Management Architecture
- **Device Discovery**: Network scanning capabilities for edge device detection
- **Connection States**: Real-time device status tracking (connected, disconnected, connecting)
- **Device Types**: Support for Raspberry Pi, NVIDIA Jetson, and other edge platforms
- **Specifications Tracking**: CPU, memory, temperature, and usage monitoring

### Chat and AI Integration
- **Multimodal Support**: Text and image inputs for AI conversations
- **Device-specific Chat**: Chat sessions tied to specific connected edge devices
- **Debug Mode**: Advanced debugging with system prompts, model inputs/outputs, and performance metrics
- **Camera Integration**: WebRTC camera capture for real-time image input
- **File Upload**: Image attachment support with client-side preview

## External Dependencies

### Core Infrastructure
- **Neon Database**: Serverless PostgreSQL hosting with connection pooling
- **Replit Platform**: Development environment with integrated deployment and domain hosting

### Frontend Libraries
- **Radix UI**: Accessible component primitives for complex UI elements
- **Tailwind CSS**: Utility-first CSS framework with custom design system
- **React Query**: Server state management with caching and synchronization
- **Wouter**: Lightweight routing library for single-page applications
- **React Hook Form**: Form validation and management with Zod schemas

### Backend Dependencies
- **Drizzle ORM**: Type-safe database operations with PostgreSQL dialect
- **Express.js**: Web application framework with middleware support
- **Multer**: File upload handling for multimodal chat features
- **Zod**: Runtime type validation for API inputs and database schemas

### Development Tools
- **Vite**: Fast build tool with HMR and TypeScript support
- **ESBuild**: JavaScript bundler for production builds
- **TypeScript**: Static type checking across frontend and backend
- **PostCSS**: CSS processing with Tailwind CSS integration

### Third-party Integrations
- **Google Fonts**: Inter and JetBrains Mono font families
- **OIDC Providers**: Google, GitHub, and Apple authentication
- **WebRTC**: Browser camera access for image capture functionality