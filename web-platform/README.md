# web-platform

React/Express management interface for the VLMChat edge AI platform. Provides real-time device monitoring, detection alerts, watchlist management, and MQTT bridging.

For the full project overview see the [root README](../README.md).

## Stack

- **Frontend**: React + TypeScript + shadcn/ui + TailwindCSS
- **Backend**: Express.js + Drizzle ORM (PostgreSQL)
- **Real-time**: Mosquitto MQTT broker (device comms) + WebSocket (browser updates)
- **Database**: PostgreSQL

## Running

All services run via Docker Compose. Run from this directory:

```bash
docker-compose up       # Start all services
docker-compose up -d    # Background
docker-compose logs -f  # View logs
docker-compose down     # Stop
```

Or from the project root using `just`:

```bash
just dev        # Start
just dev-bg     # Background
just logs       # Logs
just stop       # Stop
```

Access the app at **http://localhost:5000**.

## Local Development (without Docker)

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and MQTT broker address
   ```

3. Push database schema:
   ```bash
   npm run db:push
   ```

4. Start dev server:
   ```bash
   npm run dev
   ```

## Database

```bash
npm run db:push     # Push schema changes
npx drizzle-kit studio  # Open Drizzle Studio (GUI)
```

Schema is defined in `shared/schema.ts` (Drizzle).

## Development Commands

```bash
npm run dev       # Start with hot reload
npm run build     # Build for production
npm run check     # TypeScript type check
npm run db:push   # Push database schema
```

## API Documentation

- REST API: `/docs/openapi` (ReDoc)
- MQTT topics: `/docs/asyncapi-mqtt`
- WebSocket events: `/docs/asyncapi-websocket`

Regenerate from specs (run from project root):
```bash
just generate-docs
```