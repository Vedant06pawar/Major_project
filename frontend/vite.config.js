import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
  ],
  server: {
    proxy: {
      '/process': 'http://localhost:8000',
      '/status': 'http://localhost:8000',
      '/blocks': 'http://localhost:8000',
      '/audio': 'http://localhost:8000',
      '/export': 'http://localhost:8000',
      '/session': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
