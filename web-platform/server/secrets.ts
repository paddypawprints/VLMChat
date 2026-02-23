/**
 * Secrets Manager - Infisical Integration
 * 
 * Provides secure access to secrets stored in Infisical with fallback to environment variables.
 * All secret access is logged for audit purposes.
 */

import { InfisicalClient } from "@infisical/sdk";

interface SecretConfig {
  name: string;
  required?: boolean;
  fallbackEnvVar?: string;
}

class SecretsManager {
  private client: InfisicalClient | null = null;
  private initialized = false;
  private cache = new Map<string, { value: string; timestamp: number }>();
  private readonly CACHE_TTL = 300000; // 5 minutes
  private readonly environment = process.env.NODE_ENV || 'development';

  constructor() {
    // NOTE: Infisical integration disabled due to upstream bugs
    // Using environment variables directly with proper logging
    console.log('[Secrets] Using environment variables (Infisical disabled)');
  }

  /**
   * Get a secret by name
   * Uses environment variables directly (Infisical disabled)
   */
  async get(secretName: string, options: { required?: boolean; fallbackEnvVar?: string } = {}): Promise<string> {
    const { required = true, fallbackEnvVar } = options;

    // Check cache first
    const cached = this.cache.get(secretName);
    if (cached && (Date.now() - cached.timestamp) < this.CACHE_TTL) {
      return cached.value;
    }

    let secretValue: string | undefined;

    // Try environment variable directly
    secretValue = process.env[secretName];
    
    // Fallback to alternative env var name if provided
    if (!secretValue && fallbackEnvVar) {
      secretValue = process.env[fallbackEnvVar];
      if (secretValue) {
        console.log(`[Secrets] Using fallback env var ${fallbackEnvVar} for '${secretName}'`);
      }
    }

    // Check if required
    if (!secretValue && required) {
      throw new Error(
        `Secret '${secretName}' is required but not found in environment variables. ` +
        `Please add it to your .env file.`
      );
    }

    // Cache the result
    if (secretValue) {
      this.cache.set(secretName, { value: secretValue, timestamp: Date.now() });
      console.log(`[Secrets] Retrieved '${secretName}' from environment`);
    }

    return secretValue || '';
  }

  /**
   * Get multiple secrets at once
   */
  async getAll(secretNames: string[]): Promise<Record<string, string>> {
    const results: Record<string, string> = {};
    
    for (const name of secretNames) {
      try {
        results[name] = await this.get(name, { required: false });
      } catch (error) {
        console.error(`[Secrets] Failed to get '${name}':`, error);
      }
    }

    return results;
  }

  /**
   * Clear the cache (useful for testing or forcing refresh)
   */
  clearCache(): void {
    this.cache.clear();
    console.log('[Secrets] Cache cleared');
  }

  /**
   * Check if Infisical is available
   */
  isInfisicalAvailable(): boolean {
    return this.initialized && this.client !== null;
  }
}

// Singleton instance
export const secretsManager = new SecretsManager();

// Convenience functions
export async function getSecret(name: string, fallbackEnvVar?: string): Promise<string> {
  return secretsManager.get(name, { fallbackEnvVar });
}

export async function getSecretOptional(name: string, fallbackEnvVar?: string): Promise<string | undefined> {
  try {
    return await secretsManager.get(name, { required: false, fallbackEnvVar });
  } catch {
    return undefined;
  }
}
