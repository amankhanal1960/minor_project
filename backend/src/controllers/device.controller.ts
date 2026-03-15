import { Request, Response } from "express";
import prisma from "../prisma/client";
import { randomBytes } from "crypto";

export async function registerDevice(req: Request, res: Response) {
  try {
    const { deviceId, name, tokenHash } = req.body;

    if (!name) {
      return res.status(400).json({ error: "name is required" });
    }

    const id = deviceId ?? cryptoId();
    const hash = tokenHash ?? cryptoId(16);

    const device = await prisma.device.create({
      data: {
        id,
        name,
        tokenHash: hash,
      },
    });

    return res.status(201).json({
      success: true,
      data: device,
    });
  } catch (error: any) {
    console.error(error);
    if (error.code === "P2002") {
      return res.status(409).json({ error: "Device ID already exists" });
    }
    return res.status(500).json({ error: "Failed to register device" });
  }
}

function cryptoId(bytes = 12) {
  return randomBytes(bytes).toString("hex");
}
