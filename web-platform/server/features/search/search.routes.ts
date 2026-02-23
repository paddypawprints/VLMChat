/**
 * Search routes - Detection filtering and search term management
 */
import { Router } from 'express';
import { SearchHandler } from './search.handler';
import { requireAuth } from '../../core/middleware';

const router = Router();

// Lazy-initialize handler to avoid initialization order issues
let handler: SearchHandler;
function getHandler() {
  if (!handler) {
    handler = new SearchHandler();
  }
  return handler;
}

// Search term CRUD
router.get('/search-terms', requireAuth, (req, res, next) => getHandler().listSearchTerms(req, res, next));
router.post('/search-terms', requireAuth, (req, res, next) => getHandler().createSearchTerm(req, res, next));
router.get('/search-terms/:id', requireAuth, (req, res, next) => getHandler().getSearchTerm(req, res, next));
router.delete('/search-terms/:id', requireAuth, (req, res, next) => getHandler().deleteSearchTerm(req, res, next));

// Detection queries
router.get('/search-terms/detections', requireAuth, (req, res, next) => getHandler().getAllDetections(req, res, next));
router.get('/search-terms/:id/detections', requireAuth, (req, res, next) => getHandler().getSearchTermDetections(req, res, next));

export default router;
