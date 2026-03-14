import { Router } from "express";
import {
  createDetection,
  getDetections,
} from "../controllers/detection.controller";

const router = Router();

router.post("/detections", createDetection);
router.get("/detections", getDetections);

export default router;
