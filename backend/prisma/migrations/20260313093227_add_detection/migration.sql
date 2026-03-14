/*
  Warnings:

  - Added the required column `updatedAt` to the `Device` table without a default value. This is not possible if the table is not empty.

*/
-- CreateEnum
CREATE TYPE "DeviceStatus" AS ENUM ('ONLINE', 'OFFLINE', 'UNKNOWN');

-- CreateEnum
CREATE TYPE "Severity" AS ENUM ('LOW', 'MEDIUM', 'HIGH');

-- AlterTable
ALTER TABLE "Device" ADD COLUMN     "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "status" "DeviceStatus" NOT NULL DEFAULT 'ONLINE',
ADD COLUMN     "updatedAt" TIMESTAMP(3) NOT NULL;

-- AlterTable
ALTER TABLE "Event" ADD COLUMN     "durationMs" INTEGER,
ADD COLUMN     "fusionHit" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "motionPeakG" DOUBLE PRECISION,
ADD COLUMN     "rms" DOUBLE PRECISION,
ADD COLUMN     "severity" "Severity" NOT NULL DEFAULT 'LOW';

-- CreateTable
CREATE TABLE "Detection" (
    "id" TEXT NOT NULL,
    "deviceId" TEXT NOT NULL,
    "coughProbability" DOUBLE PRECISION NOT NULL,
    "audioLevel" DOUBLE PRECISION NOT NULL,
    "detectionAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Detection_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "idx_detection_device_time" ON "Detection"("deviceId", "detectionAt");

-- CreateIndex
CREATE INDEX "idx_event_ts" ON "Event"("ts");

-- AddForeignKey
ALTER TABLE "Detection" ADD CONSTRAINT "Detection_deviceId_fkey" FOREIGN KEY ("deviceId") REFERENCES "Device"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- RenameIndex
ALTER INDEX "Event_deviceId_ts_idx" RENAME TO "idx_event_device_ts";
