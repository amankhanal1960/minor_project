import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import router from "./routes/detection.route";

dotenv.config();

const app = express();

app.use(
  cors({
    origin: process.env.UI_ORIGIN ?? "*",
  }),
);

app.use(express.json());

app.use("/api", router);

app.get("/", (req, res) => {
  res.send("API running");
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Srver running on port ${PORT}`);
});
