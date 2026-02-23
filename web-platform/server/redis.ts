import Redis from 'ioredis';

let redisClient: Redis | null = null;

export async function setupRedis(): Promise<Redis> {
  const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
  
  redisClient = new Redis(redisUrl, {
    maxRetriesPerRequest: 3,
    retryStrategy(times) {
      const delay = Math.min(times * 50, 2000);
      return delay;
    },
    reconnectOnError(err) {
      const targetError = 'READONLY';
      if (err.message.includes(targetError)) {
        return true; // Reconnect
      }
      return false;
    },
  });

  redisClient.on('connect', () => {
    console.log('[Redis] Connected to Redis server');
  });

  redisClient.on('error', (err) => {
    console.error('[Redis] Connection error:', err);
  });

  redisClient.on('ready', () => {
    console.log('[Redis] Client ready');
  });

  // Test connection
  try {
    await redisClient.ping();
    console.log('[Redis] Connection test successful');
  } catch (error) {
    console.error('[Redis] Connection test failed:', error);
    throw error;
  }

  return redisClient;
}

export function getRedisClient(): Redis {
  if (!redisClient) {
    throw new Error('Redis client not initialized. Call setupRedis() first.');
  }
  return redisClient;
}

export async function closeRedis(): Promise<void> {
  if (redisClient) {
    await redisClient.quit();
    redisClient = null;
  }
}
