import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, boolean, json, integer } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User schema for authentication
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: text("email").notNull().unique(),
  name: text("name").notNull(),
  provider: text("provider"), // 'google', 'github', 'email', etc.
  providerId: text("provider_id"),
  createdAt: timestamp("created_at").defaultNow(),
});

// Edge device schema
export const devices = pgTable("devices", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  type: text("type").notNull(), // 'raspberry-pi', 'jetson', etc.
  ip: text("ip").notNull(),
  
  // Device authentication (PKI-based)
  publicKey: text("public_key"), // PEM-encoded Ed25519 public key (optional for development)
  keyAlgorithm: text("key_algorithm").default("Ed25519"),
  jwtVersion: integer("jwt_version").default(1), // For key rotation/revocation
  
  // Status and ownership
  status: text("status").notNull().default("manufactured"), // 'manufactured', 'registered', 'connected', 'disconnected'
  specs: json("specs"), // CPU, memory, etc.
  userId: varchar("user_id").references(() => users.id), // NULL = unclaimed
  claimedAt: timestamp("claimed_at"),
  
  // Timestamps
  manufacturedAt: timestamp("manufactured_at").defaultNow(),
  lastSeen: timestamp("last_seen"),
});

// Chat messages schema
export const chatMessages = pgTable("chat_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  deviceId: varchar("device_id").references(() => devices.id),
  role: text("role").notNull(), // 'user' or 'assistant'
  content: text("content").notNull(),
  images: json("images"), // Array of image URLs/paths
  debug: json("debug"), // Debug information
  createdAt: timestamp("created_at").defaultNow(),
});

// Device messages/events from MQTT
export const deviceMessages = pgTable("device_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  deviceId: varchar("device_id").references(() => devices.id).notNull(),
  topic: text("topic").notNull(), // MQTT topic
  payload: json("payload").notNull(), // Message payload
  qos: text("qos"), // MQTT QoS level
  retained: boolean("retained").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

// Watchlist items for detection
export const watchlistItems = pgTable("watchlist_items", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  description: text("description").notNull(),
  embeddings: json("embeddings"), // ML model embeddings
  model: text("model").default("clip-vit-base"),
  threshold: json("threshold").default(0.85), // Detection confidence threshold
  enabled: boolean("enabled").default(true),
  userId: varchar("user_id").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Detection alerts from devices
export const alerts = pgTable("alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  deviceId: varchar("device_id").references(() => devices.id).notNull(),
  watchlistItemId: varchar("watchlist_item_id").references(() => watchlistItems.id),
  type: text("type").notNull(), // 'detection', 'system', 'error'
  description: text("description").notNull(),
  confidence: json("confidence"), // Detection confidence score
  imageUrl: text("image_url"), // URL to image in storage
  metadata: json("metadata"), // Bounding box, inference time, etc.
  
  // VLM verification fields
  vlmRequired: boolean("vlm_required").default(false),
  vlmStatus: text("vlm_status"), // 'confirmed' | 'timeout' | 'invalid_response' | null
  vlmResponse: text("vlm_response"), // Raw VLM text response for debugging
  vlmInferenceTime: json("vlm_inference_time"), // Inference time in seconds
  
  createdAt: timestamp("created_at").defaultNow(),
});

// Search terms for detection filtering
export const searchTerms = pgTable("search_terms", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  searchString: text("search_string").notNull(),
  categoryMask: json("category_mask").notNull(), // boolean[80]
  categoryColors: json("category_colors").notNull(), // (string|null)[80]
  attributeMask: json("attribute_mask").notNull(), // boolean[26]
  attributeColors: json("attribute_colors").notNull(), // (string|null)[26]
  colorRequirements: json("color_requirements"), // Record<number, Record<string, string[]>> - {category_id: {region: [colors]}}
  groqResponse: json("groq_response"), // Raw Groq API response for debugging/audit
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Detection audit log for matched detections
export const detectionAuditLog = pgTable("detection_audit_log", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  searchTermId: varchar("search_term_id").references(() => searchTerms.id).notNull(),
  deviceId: varchar("device_id").references(() => devices.id).notNull(),
  timestamp: timestamp("timestamp").notNull(),
  bbox: json("bbox").notNull(), // [x1, y1, x2, y2]
  confidence: json("confidence").notNull(), // number
  category: text("category").notNull(),
  categoryId: integer("category_id").notNull(),
  attributes: json("attributes"), // PA100K attributes with confidence
  imageUrl: text("image_url"), // URL to stored image
  metadata: json("metadata"), // Additional context
  createdAt: timestamp("created_at").defaultNow(),
});

// Device configurations - per-device settings
export const deviceConfigs = pgTable("device_configs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  deviceType: varchar("device_type").notNull(), // Device platform: macos, raspberry-pi, jetson, generic
  deviceId: varchar("device_id").references(() => devices.id), // Optional: specific device override
  config: json("config").notNull(), // Full device config JSON (tasks + sinks)
  version: integer("version").notNull().default(1), // Config version for optimistic locking
  isActive: boolean("is_active").notNull().default(true), // Allow disabling without deleting
  updatedBy: varchar("updated_by").references(() => users.id), // Who made the last change
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Framework configurations - shared pipeline settings
export const frameworkConfigs = pgTable("framework_configs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull().default("default"), // Allow multiple named configs
  config: json("config").notNull(), // Full framework config JSON (pipeline + sources + buffers)
  version: integer("version").notNull().default(1), // Config version for optimistic locking
  isActive: boolean("is_active").notNull().default(true), // Only one can be active at a time
  updatedBy: varchar("updated_by").references(() => users.id), // Who made the last change
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Config history - track all changes for audit and rollback
export const configHistory = pgTable("config_history", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  configType: text("config_type").notNull(), // 'device' or 'framework'
  configId: varchar("config_id").notNull(), // References deviceConfigs.id or frameworkConfigs.id
  deviceType: varchar("device_type"), // Device platform type (for device configs)
  deviceId: varchar("device_id").references(() => devices.id), // NULL for framework configs, optional for device type configs
  config: json("config").notNull(), // Snapshot of config at this point
  version: integer("version").notNull(), // Version number
  changeDescription: text("change_description"), // Optional description of what changed
  changedBy: varchar("changed_by").references(() => users.id), // Who made the change
  createdAt: timestamp("created_at").defaultNow(),
});

// Device specs schema - standardized format
export const deviceSpecsSchema = z.object({
  cpu: z.string().optional(),
  memory: z.string().optional(),
  temperature: z.number().optional(),
  usage: z.number().optional(),
});

// Metrics command schema - sent from platform to device
export const metricsCommandSchema = z.object({
  enabled: z.boolean(),
  frequency: z.number().min(1).default(30).optional(),
  instruments: z.union([z.literal("*"), z.array(z.string())]).optional(),
});

// Metrics data schema - sent from device to platform
export const metricsDataSchema = z.object({
  timestamp: z.string(),
  session: z.object({
    start_time: z.number(),
    end_time: z.number().nullable(),
    instruments: z.array(
      z.object({
        timeseries: z.string(),
        instrument: z.object({
          type: z.string(),
          name: z.string(),
          binding_keys: z.array(z.string()).optional(),
        }).passthrough(), // Allow additional instrument-specific fields
      })
    ),
  }),
});

// Zod schemas
export const insertUserSchema = createInsertSchema(users).pick({
  email: true,
  name: true,
  provider: true,
  providerId: true,
});

export const insertDeviceSchema = createInsertSchema(devices).omit({
  userId: true,
  lastSeen: true,
});

export const insertChatMessageSchema = createInsertSchema(chatMessages).omit({
  id: true,
  createdAt: true,
});

export const insertDeviceMessageSchema = createInsertSchema(deviceMessages).omit({
  id: true,
  createdAt: true,
});

export const insertWatchlistItemSchema = createInsertSchema(watchlistItems).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertAlertSchema = createInsertSchema(alerts).omit({
  id: true,
  createdAt: true,
});

export const insertSearchTermSchema = createInsertSchema(searchTerms).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertDetectionAuditLogSchema = createInsertSchema(detectionAuditLog).omit({
  id: true,
  createdAt: true,
});

export const insertDeviceConfigSchema = createInsertSchema(deviceConfigs).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertFrameworkConfigSchema = createInsertSchema(frameworkConfigs).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertConfigHistorySchema = createInsertSchema(configHistory).omit({
  id: true,
  createdAt: true,
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type Device = typeof devices.$inferSelect;
export type InsertDevice = z.infer<typeof insertDeviceSchema>;
export type ChatMessage = typeof chatMessages.$inferSelect;
export type InsertChatMessage = z.infer<typeof insertChatMessageSchema>;
export type DeviceMessage = typeof deviceMessages.$inferSelect;
export type InsertDeviceMessage = z.infer<typeof insertDeviceMessageSchema>;
export type WatchlistItem = typeof watchlistItems.$inferSelect;
export type InsertWatchlistItem = z.infer<typeof insertWatchlistItemSchema>;
export type Alert = typeof alerts.$inferSelect;
export type InsertAlert = z.infer<typeof insertAlertSchema>;
export type SearchTerm = typeof searchTerms.$inferSelect;
export type InsertSearchTerm = z.infer<typeof insertSearchTermSchema>;
export type DetectionAuditLog = typeof detectionAuditLog.$inferSelect;
export type InsertDetectionAuditLog = z.infer<typeof insertDetectionAuditLogSchema>;
export type DeviceSpecs = z.infer<typeof deviceSpecsSchema>;
export type MetricsCommand = z.infer<typeof metricsCommandSchema>;
export type MetricsData = z.infer<typeof metricsDataSchema>;
export type DeviceConfig = typeof deviceConfigs.$inferSelect;
export type InsertDeviceConfig = z.infer<typeof insertDeviceConfigSchema>;
export type FrameworkConfig = typeof frameworkConfigs.$inferSelect;
export type InsertFrameworkConfig = z.infer<typeof insertFrameworkConfigSchema>;
export type ConfigHistory = typeof configHistory.$inferSelect;
export type InsertConfigHistory = z.infer<typeof insertConfigHistorySchema>;
