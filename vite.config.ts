import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// base is set for a GitHub Pages *project* site served under
// https://<user>.github.io/geometry-error-diagnoser/. Hosting at a domain root
// instead? Set base back to '/'.
export default defineConfig({
  base: '/geometry-error-diagnoser/',
  plugins: [react()],
})
