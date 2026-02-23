import { getRedisClient } from './redis';
import { randomUUID } from 'crypto';

const USER_SESSION_TTL = 7 * 24 * 60 * 60; // 7 days in seconds
const USER_SESSION_PREFIX = 'session:user:';
const DEVICE_SESSION_PREFIX = 'session:device:';

export interface UserSession {
  userId: string;
  createdAt: string;
}

export interface DeviceSession {
  deviceId: string;
  connectedAt: string;
  clientId: string;
}

/**
 * Create a new user session
 */
export async function createUserSession(userId: string): Promise<string> {
  const redis = getRedisClient();
  const sessionId = randomUUID();
  
  const session: UserSession = {
    userId,
    createdAt: new Date().toISOString(),
  };
  
  const key = `${USER_SESSION_PREFIX}${sessionId}`;
  await redis.setex(key, USER_SESSION_TTL, JSON.stringify(session));
  
  console.log(`[Sessions] Created user session: ${sessionId} for user: ${userId}`);
  return sessionId;
}

/**
 * Get user session by sessionId
 */
export async function getUserSession(sessionId: string): Promise<UserSession | null> {
  const redis = getRedisClient();
  const key = `${USER_SESSION_PREFIX}${sessionId}`;
  
  const data = await redis.get(key);
  if (!data) {
    return null;
  }
  
  return JSON.parse(data) as UserSession;
}

/**
 * Delete user session (logout)
 */
export async function deleteUserSession(sessionId: string): Promise<void> {
  const redis = getRedisClient();
  const key = `${USER_SESSION_PREFIX}${sessionId}`;
  
  await redis.del(key);
  console.log(`[Sessions] Deleted user session: ${sessionId}`);
}

/**
 * Create device session
 * Returns false if device already has an active session
 */
export async function createDeviceSession(deviceId: string, clientId: string): Promise<boolean> {
  const redis = getRedisClient();
  const key = `${DEVICE_SESSION_PREFIX}${deviceId}`;
  
  // Check if session already exists
  const existing = await redis.get(key);
  if (existing) {
    console.log(`[Sessions] Device ${deviceId} already has active session`);
    return false;
  }
  
  const session: DeviceSession = {
    deviceId,
    connectedAt: new Date().toISOString(),
    clientId,
  };
  
  // No TTL - cleared on disconnect or server restart
  await redis.set(key, JSON.stringify(session));
  console.log(`[Sessions] Created device session for: ${deviceId}`);
  return true;
}

/**
 * Get device session
 */
export async function getDeviceSession(deviceId: string): Promise<DeviceSession | null> {
  const redis = getRedisClient();
  const key = `${DEVICE_SESSION_PREFIX}${deviceId}`;
  
  const data = await redis.get(key);
  if (!data) {
    return null;
  }
  
  return JSON.parse(data) as DeviceSession;
}

/**
 * Delete device session
 */
export async function deleteDeviceSession(deviceId: string): Promise<void> {
  const redis = getRedisClient();
  const key = `${DEVICE_SESSION_PREFIX}${deviceId}`;
  
  await redis.del(key);
  console.log(`[Sessions] Deleted device session: ${deviceId}`);
}

/**
 * Check if device has active session
 */
export async function isDeviceConnected(deviceId: string): Promise<boolean> {
  const session = await getDeviceSession(deviceId);
  return session !== null;
}
