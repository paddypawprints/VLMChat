/**
 * Shared middleware functions
 */
import { Request, Response, NextFunction } from 'express';
import { getUserSession } from '../sessions';

/**
 * Authentication middleware
 * Extracts sessionId from Bearer token, validates against Redis
 * Attaches userId to req.user
 */
export async function requireAuth(req: Request, res: Response, next: NextFunction) {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Authentication required' });
    }
    
    const sessionId = authHeader.substring(7);
    const session = await getUserSession(sessionId);
    
    if (!session) {
      return res.status(401).json({ error: 'Invalid or expired session' });
    }
    
    // Attach userId to request for downstream handlers
    (req as any).user = { userId: session.userId };
    next();
  } catch (error) {
    console.error('Auth middleware error:', error);
    res.status(500).json({ error: 'Authentication failed' });
  }
}
