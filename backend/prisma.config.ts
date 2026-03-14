import { defineConfig, env } from "@prisma/config";
import "dotenv/config";

/**
 * Prisma 7 puts datasource connection info in prisma.config.{js,ts}
 * instead of the schema file. The CLI loads this file for migrate/generate.
 */
export default defineConfig({
  schema: "./prisma/schema.prisma",
  datasource: {
    url: env("DATABASE_URL"),
  },
});
