import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const devPort = Number(env.VITE_DEV_PORT || "5173");
  const apiProxy = env.VITE_API_PROXY || env.VITE_API_BASE || "http://127.0.0.1:9000";

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: Number.isFinite(devPort) ? devPort : 5173,
      proxy: {
        "/api": {
          target: apiProxy,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
    preview: {
      port: Number.isFinite(devPort) ? devPort : 4173,
    },
  };
});
