import { 
  type User, 
  type InsertUser, 
  type Device, 
  type InsertDevice,
  type ChatMessage,
  type InsertChatMessage 
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // User management
  getUser(id: string): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Device management
  getDevice(id: string): Promise<Device | undefined>;
  getUserDevices(userId: string): Promise<Device[]>;
  createDevice(device: InsertDevice & { userId: string }): Promise<Device>;
  updateDeviceStatus(id: string, status: string): Promise<void>;
  
  // Chat messages
  getChatMessages(userId: string, deviceId?: string): Promise<ChatMessage[]>;
  createChatMessage(message: InsertChatMessage): Promise<ChatMessage>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User> = new Map();
  private devices: Map<string, Device> = new Map();
  private chatMessages: Map<string, ChatMessage> = new Map();

  constructor() {
    // Initialize with mock data
    this.initializeMockData();
  }

  private initializeMockData() {
    // Mock devices for demo
    const mockDevices: Device[] = [
      {
        id: "rpi-001",
        name: "Raspberry Pi 4B",
        type: "raspberry-pi",
        ip: "192.168.1.100",
        status: "disconnected",
        specs: {
          cpu: "ARM Cortex-A72",
          memory: "4GB RAM",
          temperature: 45,
          usage: 23
        },
        userId: null,
        lastSeen: new Date()
      },
      {
        id: "jetson-001",
        name: "NVIDIA Jetson Nano",
        type: "jetson",
        ip: "192.168.1.101",
        status: "disconnected",
        specs: {
          cpu: "ARM Cortex-A57",
          memory: "4GB RAM",
          temperature: 52,
          usage: 67
        },
        userId: null,
        lastSeen: new Date()
      },
      {
        id: "coral-001",
        name: "Google Coral Dev Board",
        type: "coral",
        ip: "192.168.1.102",
        status: "disconnected",
        specs: {
          cpu: "ARM Cortex-A53",
          memory: "1GB RAM",
          temperature: 38,
          usage: 12
        },
        userId: null,
        lastSeen: new Date()
      }
    ];
    
    mockDevices.forEach(device => {
      this.devices.set(device.id, device);
    });
  }

  // User methods
  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByEmail(email: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.email === email,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { 
      ...insertUser, 
      provider: insertUser.provider || null,
      providerId: insertUser.providerId || null,
      id, 
      createdAt: new Date() 
    };
    this.users.set(id, user);
    return user;
  }

  // Device methods
  async getDevice(id: string): Promise<Device | undefined> {
    return this.devices.get(id);
  }

  async getUserDevices(userId: string): Promise<Device[]> {
    return Array.from(this.devices.values()).filter(
      device => device.userId === userId || device.userId === null
    );
  }

  async createDevice(device: InsertDevice & { userId: string }): Promise<Device> {
    const newDevice: Device = {
      ...device,
      status: device.status || "disconnected",
      specs: device.specs || null,
      lastSeen: new Date()
    };
    this.devices.set(device.id, newDevice);
    return newDevice;
  }

  async updateDeviceStatus(id: string, status: string): Promise<void> {
    const device = this.devices.get(id);
    if (device) {
      device.status = status;
      device.lastSeen = new Date();
      this.devices.set(id, device);
    }
  }

  // Chat message methods
  async getChatMessages(userId: string, deviceId?: string): Promise<ChatMessage[]> {
    const messages = Array.from(this.chatMessages.values())
      .filter(msg => msg.userId === userId)
      .filter(msg => !deviceId || msg.deviceId === deviceId)
      .sort((a, b) => a.createdAt!.getTime() - b.createdAt!.getTime());
    
    return messages;
  }

  async createChatMessage(message: InsertChatMessage): Promise<ChatMessage> {
    const id = randomUUID();
    const chatMessage: ChatMessage = {
      ...message,
      deviceId: message.deviceId || null,
      images: message.images || null,
      debug: message.debug || null,
      id,
      createdAt: new Date()
    };
    this.chatMessages.set(id, chatMessage);
    return chatMessage;
  }
}

export const storage = new MemStorage();
