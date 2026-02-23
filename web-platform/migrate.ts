import { sql } from 'drizzle-orm';
import { db } from './server/db.js';

async function migrate() {
  console.log('🔄 Running migrations...');
  
  try {
    // Device configurations table
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS device_configs (
        id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
        device_id VARCHAR NOT NULL UNIQUE REFERENCES devices(id),
        config JSON NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        is_active BOOLEAN NOT NULL DEFAULT true,
        updated_by VARCHAR REFERENCES users(id),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `);
    console.log('✅ device_configs table created');

    // Framework configurations table
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS framework_configs (
        id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL DEFAULT 'default',
        config JSON NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        is_active BOOLEAN NOT NULL DEFAULT true,
        updated_by VARCHAR REFERENCES users(id),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `);
    console.log('✅ framework_configs table created');

    // Config history table
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS config_history (
        id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
        config_type TEXT NOT NULL,
        config_id VARCHAR NOT NULL,
        device_id VARCHAR REFERENCES devices(id),
        config JSON NOT NULL,
        version INTEGER NOT NULL,
        change_description TEXT,
        changed_by VARCHAR REFERENCES users(id),
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);
    console.log('✅ config_history table created');

    // Create indices
    await db.execute(sql`CREATE INDEX IF NOT EXISTS idx_device_configs_device_id ON device_configs(device_id)`);
    await db.execute(sql`CREATE INDEX IF NOT EXISTS idx_device_configs_active ON device_configs(is_active) WHERE is_active = true`);
    await db.execute(sql`CREATE INDEX IF NOT EXISTS idx_framework_configs_active ON framework_configs(is_active) WHERE is_active = true`);
    await db.execute(sql`CREATE INDEX IF NOT EXISTS idx_config_history_type ON config_history(config_type)`);
    await db.execute(sql`CREATE INDEX IF NOT EXISTS idx_config_history_device_id ON config_history(device_id)`);
    await db.execute(sql`CREATE INDEX IF NOT EXISTS idx_config_history_config_id ON config_history(config_id)`);
    console.log('✅ Indices created');

    console.log('✅ Migration completed successfully');
    process.exit(0);
  } catch (error) {
    console.error('❌ Migration failed:', error);
    process.exit(1);
  }
}

migrate();
