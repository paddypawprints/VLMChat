/**
 * Seed default framework configuration
 * Run with: npx tsx server/db/seed-framework-config.ts
 */

import { db } from '../db';
import { frameworkConfigs, users } from '@shared/schema';
import { eq } from 'drizzle-orm';

const defaultFrameworkConfig = {
  pipeline: {
    max_workers: 4,
    enable_trace: true,
    buffer_size: 100
  },
  sources: {
    camera: {
      fps: 30,
      width: 1920,
      height: 1080,
      flip_method: 2
    }
  },
  buffers: {
    ring_buffer_size: 300,
    pool_size: 60
  }
};

async function seed() {
  console.log('Seeding framework configuration...');
  
  // Check if framework config already exists
  const existing = await db
    .select()
    .from(frameworkConfigs)
    .where(eq(frameworkConfigs.isActive, true))
    .limit(1);
  
  if (existing.length > 0) {
    console.log('Framework configuration already exists, skipping seed');
    return;
  }
  
  // Get first user (or create system user if needed)
  let [user] = await db.select().from(users).limit(1);
  if (!user) {
    [user] = await db.insert(users).values({
      email: 'system@vlmchat.local',
      name: 'System',
      provider: 'system'
    }).returning();
  }
  
  // Create default framework config
  const [config] = await db
    .insert(frameworkConfigs)
    .values({
      name: 'default',
      config: defaultFrameworkConfig,
      version: 1,
      isActive: true,
      updatedBy: user.id,
    })
    .returning();
  
  console.log('✓ Created framework configuration:', config.id);
}

seed()
  .then(() => {
    console.log('✓ Seed complete');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Seed error:', error);
    process.exit(1);
  });
