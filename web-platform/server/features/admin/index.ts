import { Router } from 'express';
import { requireAuth } from '../../core/middleware';
import * as adminHandler from './admin.handler';

const router = Router();

// Device management routes
router.get('/admin/devices', requireAuth, adminHandler.listDevices);
router.post('/admin/devices', requireAuth, adminHandler.createDevice);
router.patch('/admin/devices/:deviceId', requireAuth, adminHandler.updateDevice);
router.delete('/admin/devices/:deviceId', requireAuth, adminHandler.deleteDevice);

// Service management routes
router.get('/admin/services', requireAuth, adminHandler.listServices);
router.post('/admin/services', requireAuth, adminHandler.createService);
router.patch('/admin/services/:serviceId', requireAuth, adminHandler.updateService);
router.delete('/admin/services/:serviceId', requireAuth, adminHandler.deleteService);

export { router as adminRoutes };
