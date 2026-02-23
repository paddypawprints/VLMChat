import { Router } from 'express';
import { requireAuth } from '../../core/middleware';
import {
  getDeviceConfig,
  updateDeviceConfig,
  getDeviceConfigHistory,
  rollbackDeviceConfig,
  getFrameworkConfig,
  updateFrameworkConfig,
  getFrameworkConfigHistory,
} from './config.handler';

const router = Router();

// Device configuration routes (by device type, not individual device)
router.get('/config/device', requireAuth, getDeviceConfig);
router.put('/config/device', requireAuth, updateDeviceConfig);
router.get('/config/device/history', requireAuth, getDeviceConfigHistory);
router.post('/config/device/rollback/:historyId', requireAuth, rollbackDeviceConfig);

// Framework configuration routes
router.get('/config/framework', requireAuth, getFrameworkConfig);
router.put('/config/framework', requireAuth, updateFrameworkConfig);
router.get('/config/framework/history', requireAuth, getFrameworkConfigHistory);

export { router as configRoutes };
