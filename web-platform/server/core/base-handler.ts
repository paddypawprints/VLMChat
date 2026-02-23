/**
 * Base handler class providing cross-cutting concerns for all route handlers
 */
import { Request, Response, NextFunction } from 'express';
import { db } from '../db';
import { getRedisClient } from '../redis';

export interface HandlerOptions {
  /** Enable request validation against OpenAPI spec (default: true in dev, false in prod) */
  validateRequest?: boolean;
  /** Enable response validation against OpenAPI spec (default: true in dev, false in prod) */
  validateResponse?: boolean;
  /** Log request/response details (default: false) */
  enableLogging?: boolean;
}

export abstract class BaseHandler {
  protected db = db;
  
  /** Lazy-load Redis client to avoid initialization order issues */
  protected get redis() {
    return getRedisClient();
  }
  
  /** Default handler options */
  protected defaultOptions: HandlerOptions = {
    validateRequest: process.env.NODE_ENV === 'development',
    validateResponse: process.env.NODE_ENV === 'development',
    enableLogging: false,
  };

  /**
   * Wraps handler with error handling, logging, validation, and lifecycle hooks
   */
  protected handleRequest(
    handler: (req: Request, res: Response) => Promise<void>,
    options?: HandlerOptions
  ) {
    const opts = { ...this.defaultOptions, ...options };
    
    return async (req: Request, res: Response, next: NextFunction) => {
      const startTime = Date.now();
      
      try {
        // Request validation happens via express-openapi-validator middleware
        // It runs before this handler, so requests are already validated
        
        if (opts.enableLogging) {
          console.log(`[${this.constructor.name}] ${req.method} ${req.path}`, {
            query: req.query,
            body: req.body,
          });
        }
        
        await this.beforeRequest(req, res);
        await handler.call(this, req, res);
        await this.afterRequest(req, res, startTime);
        
        if (opts.enableLogging) {
          const duration = Date.now() - startTime;
          console.log(`[${this.constructor.name}] Response sent in ${duration}ms`);
        }
      } catch (error) {
        await this.onError(error, req, res);
      }
    };
  }

  /**
   * Hook: Called before request processing
   * Override in subclass for feature-specific setup
   */
  protected async beforeRequest(req: Request, res: Response): Promise<void> {
    // Default: no-op
    // Subclasses can override for logging, validation, etc.
  }

  /**
   * Hook: Called after successful request processing
   * Override in subclass for feature-specific cleanup
   */
  protected async afterRequest(req: Request, res: Response, startTime: number): Promise<void> {
    // Default: no-op
    // Subclasses can override for metrics, audit logging, etc.
  }

  /**
   * Hook: Called on error during request processing
   * Override in subclass for custom error handling
   */
  protected async onError(error: any, req: Request, res: Response): Promise<void> {
    console.error(`[${this.constructor.name}] Error:`, error);
    
    // OpenAPI validation errors (from express-openapi-validator)
    if (error.status === 400 && error.errors) {
      return res.status(400).json({
        error: 'Request validation failed',
        details: error.errors,
      });
    }
    
    const status = error.status || error.statusCode || 500;
    const message = error.message || 'Internal server error';
    
    res.status(status).json({ error: message });
  }

  /**
   * Utility: Extract authenticated user ID from request
   */
  protected getUserId(req: Request): string {
    const userId = (req as any).user?.userId;
    if (!userId) {
      throw new Error('User not authenticated');
    }
    return userId;
  }

  /**
   * Utility: Validate required fields in request body
   */
  protected validateRequired(body: any, fields: string[]): void {
    const missing = fields.filter(field => !body[field]);
    if (missing.length > 0) {
      const error: any = new Error(`Missing required fields: ${missing.join(', ')}`);
      error.status = 400;
      throw error;
    }
  }

  
  /**
   * Utility: Validate response against expected schema
   * Use in development to catch response format issues
   */
  protected validateResponseSchema(data: any, expectedFields: string[]): void {
    if (process.env.NODE_ENV !== 'development') return;
    
    const missing = expectedFields.filter(field => !(field in data));
    if (missing.length > 0) {
      console.warn(`[${this.constructor.name}] Response missing fields: ${missing.join(', ')}`);
    }
  }
  /**
   * Utility: Create error with status code
   */
  protected createError(message: string, status: number = 500): Error {
    const error: any = new Error(message);
    error.status = status;
    return error;
  }
}
