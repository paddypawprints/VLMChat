import { Request, Response } from 'express';
import { db } from '../../db';
import { 
  deviceConfigs, 
  frameworkConfigs, 
  configHistory,
  devices,
  InsertDeviceConfig,
  InsertFrameworkConfig,
  InsertConfigHistory 
} from '@shared/schema';
import { eq, and, desc } from 'drizzle-orm';
import { publishToDevice } from '../../mqtt';

/**
 * Get device configuration
 * GET /api/config/device?type=macos
 */
export async function getDeviceConfig(req: Request, res: Response) {
  try {
    const userId = (req as any).user.userId;
    const { type: deviceType } = req.query;

    if (!deviceType || typeof deviceType !== 'string') {
      return res.status(400).json({ error: 'Device type is required' });
    }

    // Get active config for this device type
    const [config] = await db
      .select()
      .from(deviceConfigs)
      .where(
        and(
          eq(deviceConfigs.deviceType, deviceType),
          eq(deviceConfigs.isActive, true)
        )
      )
      .orderBy(desc(deviceConfigs.updatedAt))
      .limit(1);

    if (!config) {
      return res.status(404).json({ error: `No configuration found for device type: ${deviceType}` });
    }

    res.json(config);
  } catch (error) {
    console.error('Get device config error:', error);
    res.status(500).json({ error: 'Failed to get device configuration' });
  }
}

/**
 * Update device configuration
 * PUT /api/config/device?type=macos
 */
export async function updateDeviceConfig(req: Request, res: Response) {
  try {
    const userId = (req as any).user.userId;
    const { type: deviceType } = req.query;
    const { config: configData, changeDescription } = req.body;

    if (!deviceType || typeof deviceType !== 'string') {
      return res.status(400).json({ error: 'Device type is required' });
    }

    // Get current config for version check
    const [currentConfig] = await db
      .select()
      .from(deviceConfigs)
      .where(
        and(
          eq(deviceConfigs.deviceType, deviceType),
          eq(deviceConfigs.isActive, true)
        )
      )
      .orderBy(desc(deviceConfigs.updatedAt))
      .limit(1);

    const newVersion = currentConfig ? currentConfig.version + 1 : 1;

    // Create new config version
    const [newConfig] = await db
      .insert(deviceConfigs)
      .values({
        deviceType,
        deviceId: null, // No specific device, applies to all of this type
        config: configData,
        version: newVersion,
        isActive: true,
        updatedBy: userId,
      })
      .returning();

    // Deactivate old config if exists
    if (currentConfig) {
      await db
        .update(deviceConfigs)
        .set({ isActive: false })
        .where(eq(deviceConfigs.id, currentConfig.id));
    }

    // Save to history
    await db.insert(configHistory).values({
      configType: 'device',
      configId: newConfig.id,
      deviceId: null,
      deviceType,
      config: configData,
      version: newVersion,
      changeDescription,
      changedBy: userId,
    });

    // TODO: Publish config to all devices of this type via MQTT
    // await publishToDeviceType(deviceType, 'commands/config', {
    //   version: newVersion,
    //   config: configData,
    // });

    res.json(newConfig);
  } catch (error) {
    console.error('Update device config error:', error);
    res.status(500).json({ error: 'Failed to update device configuration' });
  }
}

/**
 * Get device configuration history
 * GET /api/config/device/history?type=macos
 */
export async function getDeviceConfigHistory(req: Request, res: Response) {
  try {
    const userId = (req as any).user.userId;
    const { type: deviceType, limit = 10 } = req.query;

    if (!deviceType || typeof deviceType !== 'string') {
      return res.status(400).json({ error: 'Device type is required' });
    }

    const history = await db
      .select()
      .from(configHistory)
      .where(
        and(
          eq(configHistory.configType, 'device'),
          eq(configHistory.deviceType, deviceType)
        )
      )
      .orderBy(desc(configHistory.createdAt))
      .limit(Number(limit));

    res.json(history);
  } catch (error) {
    console.error('Get device config history error:', error);
    res.status(500).json({ error: 'Failed to get configuration history' });
  }
}

/**
 * Rollback device configuration to a previous version
 * POST /api/config/device/rollback/:historyId?type=macos
 */
export async function rollbackDeviceConfig(req: Request, res: Response) {
  try {
    const userId = (req as any).user.userId;
    const { historyId } = req.params;
    const { type: deviceType } = req.query;

    if (!deviceType || typeof deviceType !== 'string') {
      return res.status(400).json({ error: 'Device type is required' });
    }

    // Get the historical config
    const [historicalConfig] = await db
      .select()
      .from(configHistory)
      .where(
        and(
          eq(configHistory.id, historyId),
          eq(configHistory.deviceType, deviceType)
        )
      );

    if (!historicalConfig) {
      return res.status(404).json({ error: 'Historical configuration not found' });
    }

    // Get current config for versioning
    const [currentConfig] = await db
      .select()
      .from(deviceConfigs)
      .where(
        and(
          eq(deviceConfigs.deviceType, deviceType),
          eq(deviceConfigs.isActive, true)
        )
      )
      .orderBy(desc(deviceConfigs.updatedAt))
      .limit(1);

    const newVersion = currentConfig ? currentConfig.version + 1 : 1;

    // Create new config version with rolled back data
    const [newConfig] = await db
      .insert(deviceConfigs)
      .values({
        deviceType,
        deviceId: null,
        config: historicalConfig.config,
        version: newVersion,
        isActive: true,
        updatedBy: userId,
      })
      .returning();

    // Deactivate old config
    if (currentConfig) {
      await db
        .update(deviceConfigs)
        .set({ isActive: false })
        .where(eq(deviceConfigs.id, currentConfig.id));
    }

    // Save to history
    await db.insert(configHistory).values({
      configType: 'device',
      configId: newConfig.id,
      deviceId: null,
      deviceType,
      config: historicalConfig.config,
      version: newVersion,
      changeDescription: `Rolled back to version ${historicalConfig.version}`,
      changedBy: userId,
    });

    // TODO: Publish config to all devices of this type via MQTT

    res.json(newConfig);
  } catch (error) {
    console.error('Rollback device config error:', error);
    res.status(500).json({ error: 'Failed to rollback configuration' });
  }
}

/**
 * Get framework configuration (shared pipeline settings)
 * GET /api/config/framework
 */
export async function getFrameworkConfig(req: Request, res: Response) {
  try {
    const userId = (req as any).user.userId;

    // Get active framework config
    const [config] = await db
      .select()
      .from(frameworkConfigs)
      .where(eq(frameworkConfigs.isActive, true))
      .orderBy(desc(frameworkConfigs.updatedAt))
      .limit(1);

    if (!config) {
      return res.status(404).json({ error: 'No framework configuration found' });
    }

    res.json(config);
  } catch (error) {
    console.error('Get framework config error:', error);
    res.status(500).json({ error: 'Failed to get framework configuration' });
  }
}

/**
 * Update framework configuration
 * PUT /api/config/framework
 */
export async function updateFrameworkConfig(req: Request, res: Response) {
  try {
    const userId = (req as any).user.userId;
    const { config: configData, name = 'default', changeDescription } = req.body;

    // Get current active config
    const [currentConfig] = await db
      .select()
      .from(frameworkConfigs)
      .where(eq(frameworkConfigs.isActive, true))
      .orderBy(desc(frameworkConfigs.updatedAt))
      .limit(1);

    const newVersion = currentConfig ? currentConfig.version + 1 : 1;

    // Create new config version
    const [newConfig] = await db
      .insert(frameworkConfigs)
      .values({
        name,
        config: configData,
        version: newVersion,
        isActive: true,
        updatedBy: userId,
      })
      .returning();

    // Deactivate old config
    if (currentConfig) {
      await db
        .update(frameworkConfigs)
        .set({ isActive: false })
        .where(eq(frameworkConfigs.id, currentConfig.id));
    }

    // Save to history
    await db.insert(configHistory).values({
      configType: 'framework',
      configId: newConfig.id,
      deviceId: null, // Framework configs are not device-specific
      config: configData,
      version: newVersion,
      changeDescription,
      changedBy: userId,
    });

    res.json(newConfig);
  } catch (error) {
    console.error('Update framework config error:', error);
    res.status(500).json({ error: 'Failed to update framework configuration' });
  }
}

/**
 * Get framework configuration history
 * GET /api/config/framework/history
 */
export async function getFrameworkConfigHistory(req: Request, res: Response) {
  try {
    const userId = (req as any).user.userId;
    const { limit = 10 } = req.query;

    const history = await db
      .select()
      .from(configHistory)
      .where(eq(configHistory.configType, 'framework'))
      .orderBy(desc(configHistory.createdAt))
      .limit(Number(limit));

    res.json(history);
  } catch (error) {
    console.error('Get framework config history error:', error);
    res.status(500).json({ error: 'Failed to get configuration history' });
  }
}
