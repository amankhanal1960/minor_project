import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import detectionRouter from "./routes/detection.route";
import deviceRouter from "./routes/device.route";

dotenv.config();

const app = express();

app.use(
  cors({
    origin: process.env.UI_ORIGIN ?? "*",
  }),
);

app.use(express.json());

app.use("/api", detectionRouter);
app.use("/api", deviceRouter);

app.get("/", (req, res) => {
  res.send("API running");
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Srver running on port ${PORT}`);
});
