/**
 * Search handler for managing search terms and detection filtering
 */
import { Request, Response } from 'express';
import { BaseHandler } from '../../core/base-handler';
import { searchTerms, detectionAuditLog, devices as devicesTable } from '@shared/schema';
import { eq, and, desc } from 'drizzle-orm';
import { parseWithGroq } from './groq.service';

export class SearchHandler extends BaseHandler {
  /**
   * Transform database search term to API response format
   */
  private transformSearchTerm(term: any) {
    return {
      id: term.id,
      user_id: term.userId,
      search_string: term.searchString,
      category_mask: term.categoryMask,
      category_colors: term.categoryColors,
      attribute_mask: term.attributeMask,
      attribute_colors: term.attributeColors,
      color_requirements: term.colorRequirements,
      groq_response: term.groqResponse,
      created_at: term.createdAt,
      updated_at: term.updatedAt,
    };
  }

  /**
   * Transform detection audit log entry to API response format
   */
  private transformDetection(detection: any) {
    return {
      id: detection.id,
      search_term_id: detection.searchTermId,
      device_id: detection.deviceId,
      timestamp: detection.timestamp,
      bbox: detection.bbox,
      confidence: detection.confidence,
      category: detection.category,
      category_id: detection.categoryId,
      attributes: detection.attributes,
      image_url: detection.imageUrl,
      metadata: detection.metadata,
      created_at: detection.createdAt,
    };
  }

  /**
   * GET /api/search-terms - List all search terms for authenticated user
   */
  listSearchTerms = this.handleRequest(async (req: Request, res: Response) => {
    const userId = this.getUserId(req);
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;
    
    const terms = await this.db.select()
      .from(searchTerms)
      .where(eq(searchTerms.userId, userId))
      .orderBy(desc(searchTerms.createdAt))
      .limit(limit)
      .offset(offset);
    
    res.json(terms.map(t => this.transformSearchTerm(t)));
  });

  /**
   * POST /api/search-terms - Create new search term with NLP parsing
   */
  createSearchTerm = this.handleRequest(async (req: Request, res: Response) => {
    const userId = this.getUserId(req);
    const { searchString } = req.body;
    
    this.validateRequired(req.body, ['searchString']);
    
    // Parse natural language to filter vectors using Groq API
    const filter = await parseWithGroq(searchString);
    
    // Insert into database
    const [newTerm] = await this.db.insert(searchTerms).values({
      userId,
      searchString,
      categoryMask: filter.categoryMask,
      categoryColors: filter.categoryColors,
      attributeMask: filter.attributeMask,
      attributeColors: filter.attributeColors,
      colorRequirements: filter.colorRequirements,
      groqResponse: filter.groqResponse,
    }).returning();
    
    // Publish updated filter list to all devices via MQTT
    await this.publishFilterUpdate(userId);
    
    res.status(201).json(this.transformSearchTerm(newTerm));
  });

  /**
   * GET /api/search-terms/:id - Get specific search term
   */
  getSearchTerm = this.handleRequest(async (req: Request, res: Response) => {
    const userId = this.getUserId(req);
    const { id } = req.params;
    
    const [term] = await this.db.select()
      .from(searchTerms)
      .where(and(
        eq(searchTerms.id, id),
        eq(searchTerms.userId, userId)
      ));
    
    if (!term) {
      throw this.createError('Search term not found', 404);
    }
    
    res.json(this.transformSearchTerm(term));
  });

  /**
   * DELETE /api/search-terms/:id - Delete search term
   */
  deleteSearchTerm = this.handleRequest(async (req: Request, res: Response) => {
    const userId = this.getUserId(req);
    const { id } = req.params;
    
    const [deleted] = await this.db.delete(searchTerms)
      .where(and(
        eq(searchTerms.id, id),
        eq(searchTerms.userId, userId)
      ))
      .returning();
    
    if (!deleted) {
      throw this.createError('Search term not found', 404);
    }

    // Notify devices so they stop looking for this term
    await this.publishFilterUpdate(userId);

    res.json({ success: true, id: deleted.id });
  });

  /**
   * GET /api/search-terms/detections - Get all detections across all search terms
   */
  getAllDetections = this.handleRequest(async (req: Request, res: Response) => {
    const userId = this.getUserId(req);
    const limit = parseInt(req.query.limit as string) || 100;
    const offset = parseInt(req.query.offset as string) || 0;
    
    // Get user's search term IDs
    const userTerms = await this.db.select({ id: searchTerms.id })
      .from(searchTerms)
      .where(eq(searchTerms.userId, userId));
    
    if (userTerms.length === 0) {
      throw this.createError('Search term not found', 404);
    }
    
    const termIds = userTerms.map(t => t.id);
    
    // Get detections for these search terms
    const detections = await this.db.select()
      .from(detectionAuditLog)
      .where(eq(detectionAuditLog.searchTermId, termIds[0])) // TODO: Fix to use IN clause
      .orderBy(desc(detectionAuditLog.timestamp))
      .limit(limit)
      .offset(offset);
    
    res.json({
      detections: detections.map(d => this.transformDetection(d))
    });
  });

  /**
   * Publish complete filter list to devices
   */
  async publishFilterUpdate(userId: string, deviceId?: string): Promise<void> {
    const mqttClient = (global as any).mqttClient;
    if (!mqttClient) {
      console.warn(`[${this.constructor.name}] MQTT client not available for filter update`);
      return;
    }

    // Get only this user's search terms — never publish another user's filters
    const allTerms = await this.db
      .select()
      .from(searchTerms)
      .where(eq(searchTerms.userId, userId))
      .orderBy(desc(searchTerms.createdAt));

    const filters = allTerms.map(term => {
      const groq = (term.groqResponse as any) || {};
      const strategy = groq.strategy || {};
      return {
        id: term.id,
        name: term.searchString,
        category_mask: term.categoryMask,
        category_colors: term.categoryColors,
        attribute_mask: term.attributeMask,
        attribute_colors: term.attributeColors,
        color_requirements: term.colorRequirements || {},
        vlm_required: strategy.requires_vlm ?? false,
        vlm_reasoning: strategy.vlm_reasoning ?? '',
      };
    });

    const filterList = { filters };

    if (deviceId) {
      // Publish to specific device
      const topic = `devices/${deviceId}/commands/filter`;
      mqttClient.publish(topic, JSON.stringify(filterList), { qos: 1 });
      console.log(`[${this.constructor.name}] Published ${filters.length} filters to device ${deviceId}`);
    } else {
      // Publish only to devices owned by this user
      const userDevices = await this.db
        .select()
        .from(devicesTable)
        .where(eq(devicesTable.userId, userId));
      for (const device of userDevices) {
        const topic = `devices/${device.id}/commands/filter`;
        mqttClient.publish(topic, JSON.stringify(filterList), { qos: 1 });
      }
      console.log(`[${this.constructor.name}] Published ${filters.length} filters to ${userDevices.length} devices (user ${userId})`);
    }
  }

  /**
   * GET /api/search-terms/:id/detections - Get detections for specific search term
   */
  getSearchTermDetections = this.handleRequest(async (req: Request, res: Response) => {
    const userId = this.getUserId(req);
    const { id } = req.params;
    const limit = parseInt(req.query.limit as string) || 100;
    const offset = parseInt(req.query.offset as string) || 0;
    
    // Verify ownership
    const [term] = await this.db.select()
      .from(searchTerms)
      .where(and(
        eq(searchTerms.id, id),
        eq(searchTerms.userId, userId)
      ));
    
    if (!term) {
      throw this.createError('Search term not found', 404);
    }
    
    // Get detections
    const detections = await this.db.select()
      .from(detectionAuditLog)
      .where(eq(detectionAuditLog.searchTermId, id))
      .orderBy(desc(detectionAuditLog.timestamp))
      .limit(limit)
      .offset(offset);
    
    res.json({
      detections: detections.map(d => this.transformDetection(d))
    });
  });
}
