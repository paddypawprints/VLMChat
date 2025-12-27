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
  publicKey: text("public_key").notNull(), // PEM-encoded Ed25519 public key
  keyAlgorithm: text("key_algorithm").notNull().default("Ed25519"),
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
  createdAt: timestamp("created_at").defaultNow(),
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
