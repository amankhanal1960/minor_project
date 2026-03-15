import { Router } from "express";
import { registerDevice } from "../controllers/device.controller";

const router = Router();

router.post("/devices", registerDevice);

export default router;
