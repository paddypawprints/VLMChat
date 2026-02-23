# Infisical Secrets Manager Setup

This project uses [Infisical](https://infisical.com) for secure secrets management. API keys and other sensitive data are stored encrypted in Infisical instead of environment variables or .env files.

## Quick Start

### 1. Generate Encryption Keys

```bash
# Generate encryption key (64 char hex)
openssl rand -hex 32

# Generate auth secret (32+ char string)
openssl rand -base64 32
```

Add these to your `.env` file:

```bash
INFISICAL_ENCRYPTION_KEY=<64-char-hex-from-above>
INFISICAL_AUTH_SECRET=<32-char-string-from-above>
```

### 2. Start Infisical

```bash
docker-compose up -d infisical
```

### 3. Access Infisical UI

Open http://localhost:8080 in your browser.

**First time setup:**
1. Click "Create Account"
2. Enter email and password
3. Create organization: "VLMChat"
4. Create project: "VLMChat Production"

### 4. Add Secrets via UI

In the Infisical dashboard:

1. Select your project
2. Choose environment: `development` (or `production`)
3. Click **"+ Add Secret"**
4. Add the following secrets:

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `GROQ_API_KEY` | Groq API key for NLP parsing | `gsk_...` |
| `OPENAI_API_KEY` | OpenAI API key (optional) | `sk-...` |

### 5. Generate Service Token

For your Node.js backend to access secrets:

1. In Infisical UI, go to **Project Settings → Service Tokens**
2. Click **"Create Service Token"**
3. Name: `web-backend`
4. Environment: `development`
5. Expiration: Never (or set as needed)
6. Click **Generate**
7. **Copy the token** (you won't see it again!)

### 6. Configure Backend

Add the service token to your `.env`:

```bash
INFISICAL_TOKEN=st.xxxxx.yyyyy.zzzzz
```

**Important:** This is the ONLY secret that lives in `.env` - all other secrets go in Infisical.

### 7. Restart Services

```bash
docker-compose restart web
```

### 8. Verify Setup

Check the logs to confirm secrets are loading:

```bash
docker-compose logs web | grep Secrets
```

You should see:
```
[Secrets] Infisical client initialized
[Secrets] Retrieved 'GROQ_API_KEY' from Infisical
```

## Development Workflow

### Adding a New Secret

**Option A: Via UI (Recommended)**
1. Open http://localhost:8080
2. Select your project and environment
3. Click "+ Add Secret"
4. Enter name and value
5. Save

**Option B: Via CLI**
```bash
npx infisical secrets set GROQ_API_KEY=gsk_xxxxx --env=development
```

### Using Secrets in Code

```typescript
import { getSecret } from './server/secrets';

// Get secret (throws if not found)
const groqApiKey = await getSecret('GROQ_API_KEY');

// Get secret with fallback to env var
const apiKey = await getSecret('GROQ_API_KEY', 'GROQ_API_KEY');

// Get optional secret
const optional = await getSecretOptional('OPTIONAL_KEY');
```

### Rotating Secrets

1. Update secret value in Infisical UI
2. Wait 5 minutes for cache to expire, or restart service:
   ```bash
   docker-compose restart web
   ```

## Security Best Practices

### ✅ DO:
- Store ALL sensitive data in Infisical
- Use strong encryption keys (generated with openssl)
- Limit service token permissions to specific environments
- Rotate service tokens periodically
- Use different projects for dev/staging/prod

### ❌ DON'T:
- Commit `.env` to git (it's gitignored)
- Store secrets in code or comments
- Share service tokens via chat/email
- Use the same secrets across environments

## Troubleshooting

### "Secret not found" error

1. Check secret exists in Infisical UI
2. Verify environment matches (development/production)
3. Confirm service token has access to that environment
4. Clear cache: restart web service

### Infisical UI not loading

```bash
# Check service status
docker-compose ps infisical

# View logs
docker-compose logs infisical

# Ensure DB and Redis are healthy
docker-compose ps db redis
```

### Service token expired

1. Generate new token in Infisical UI
2. Update `INFISICAL_TOKEN` in `.env`
3. Restart web service

## Production Deployment

### Environment-Specific Secrets

Create separate projects or environments for each stage:

- **Development**: Local testing
- **Staging**: Pre-production testing
- **Production**: Live environment

Use different service tokens with restricted access for each.

### High Availability

For production, run Infisical with:
- Persistent PostgreSQL (not Docker volume)
- Redis with persistence enabled
- Regular backups of Infisical database
- HTTPS for Infisical UI (reverse proxy with Let's Encrypt)

## Additional Resources

- [Infisical Documentation](https://infisical.com/docs)
- [SDK Reference](https://infisical.com/docs/sdks/overview)
- [Security Best Practices](https://infisical.com/docs/security)
