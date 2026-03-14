import { Request, Response } from "express";
import prisma from "../prisma/client";

export async function createDetection(req: Request, res: Response) {
  try {
    const { deviceId, coughProbability, audioLevel } = req.body;

    const detection = await prisma.detection.create({
      data: {
        deviceId,
        coughProbability,
        audioLevel,
      },
    });

    res.json({
      success: true,
      data: detection,
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      error: "Failed to create detection",
    });
  }
}

export async function getDetections(req: Request, res: Response) {
  try {
    const { deviceId } = req.query;

    const detections = await prisma.detection.findMany({
      where: deviceId ? { deviceId: String(deviceId) } : undefined,
      orderBy: { detectionAt: "desc" },
    });

    res.json(detections);
  } catch (error) {
    console.error(error);

    res.status(500).json({
      error: "Failed to fetch detections",
    });
  }
}
