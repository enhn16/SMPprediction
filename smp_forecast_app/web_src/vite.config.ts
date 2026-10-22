// 빌드 결과 = ../web/index.html 한 파일 (JS·CSS 포함). 데이터는 ../web/data/forecast_data.js를 그대로 읽음 (매일 운영이 갱신)
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  base: './',
  publicDir: '../web', // 개발 서버에서 data/forecast_data.js 제공
  build: { outDir: '../web', emptyOutDir: false, copyPublicDir: false },
})
