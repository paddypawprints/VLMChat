# Independent Research Edge AI Platform

A comprehensive edge AI platform that enables deployment of large language models (LLMs) and vision-language models (VLMs) directly on edge devices like Raspberry Pi and NVIDIA Jetson.

## Features

- **Edge AI Processing**: Run AI models locally without cloud dependencies
- **Zero Incremental Cost**: Fixed hardware cost with unlimited AI processing
- **Natural Language Configuration**: Configure VLMs with simple English instructions
- **Multi-device Support**: Raspberry Pi, NVIDIA Jetson, Intel NUC, and more
- **Web-based Management**: Device management and chat interface
- **Real-time Chat**: Chat with AI models running on your edge devices

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OR: Node.js 18+, Python 3.11+, PostgreSQL

### Running with Docker Compose (Recommended)

```bash
# Start all services
docker-compose up

# Access the application at http://localhost:5000
```

### Running Locally

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

3. Start PostgreSQL (if not using Docker)

4. Push database schema:
   ```bash
   npm run db:push
   ```

5. Start the AI service (in a separate terminal):
   ```bash
   python3 ai_service.py
   ```

6. Start the web server:
   ```bash
   npm run dev
   ```

7. Access at http://localhost:5000

## Architecture

The application uses a microservices architecture:

- **web** (Node.js/Express): Frontend server, API routes, database operations, authentication
- **ai-service** (Python/FastAPI): Stateless AI inference service
- **db** (PostgreSQL): Data persistence

```
┌─────────────────────┐
│  Web Service        │  ← React frontend + Express API
│  (Node.js)          │  ← Auth, CRUD, DB operations
│  Port 5000          │
└──────────┬──────────┘
           │ HTTP
           ↓
┌─────────────────────┐
│  AI Service         │  ← AI model inference only
│  (Python/FastAPI)   │  ← Stateless, no database
│  Port 8000          │
└─────────────────────┘
           
┌─────────────────────┐
│  PostgreSQL         │  ← Data storage
│  Port 5432          │
└─────────────────────┘
```

## Development

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run check` - Type check
- `npm run db:push` - Push database schema changes

## Cost Benefits

Traditional cloud AI services charge per request, leading to costs that scale linearly with usage. Edge AI provides:

- **Fixed Costs**: Pay only for hardware, not per API call
- **Unlimited Usage**: Make as many AI calls as needed
- **No Cloud Dependencies**: Works offline and with poor connectivity
- **Data Privacy**: All processing stays on your devices

## Documentation

- **Technical Whitepaper**: See `whitepaper.html` for comprehensive analysis
- **API Documentation**: Available at `/docs` when running
- **Installation Guide**: See `install.sh` for detailed setup instructions

## Support

For support and questions, contact: contact@independent-research.com

## License

MIT License - see LICENSE file for details.