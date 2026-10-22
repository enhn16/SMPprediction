import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

const root = document.getElementById('root')!
if (!window.SMP_DATA) {
  root.innerHTML = '<p style="padding:24px">데이터 파일(data/forecast_data.js)이 없습니다. python -m smp_forecast_app.src.export_web 실행 후 다시 여세요.</p>'
} else {
  createRoot(root).render(<StrictMode><App /></StrictMode>)
}
