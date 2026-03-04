# Heart Rate Estimation Web App

Ứng dụng web ước tính nhịp tim từ video khuôn mặt sử dụng Deep Learning.

## Tính năng

1. **Upload Video**: Tải video lên và phân tích nhịp tim
2. **Real-time**: Đo nhịp tim trực tiếp qua webcam với hướng dẫn đặt khuôn mặt

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + Vite |
| Styling | TailwindCSS |
| Camera | react-webcam |
| Face Detection | face-api.js (TensorFlow.js) |
| Backend | FastAPI (Python) |
| ML Model | PyTorch (TS-CST Net) |
| Communication | REST API + WebSocket |

## Cấu trúc

```
web_app/
├── frontend/                  # React Application
│   ├── src/
│   │   ├── components/       # UI Components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API services
│   │   ├── hooks/           # Custom hooks
│   │   └── styles/          # CSS styles
│   ├── public/              # Static assets
│   └── package.json
│
├── backend/                  # FastAPI Backend
│   ├── app/
│   │   ├── main.py         # API endpoints
│   │   ├── model_service.py # ML inference
│   │   ├── preprocessing.py # Video processing
│   │   └── websocket.py    # Real-time connection
│   ├── models/             # Trained models
│   └── requirements.txt
│
└── docs/                    # Documentation
```

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python run_server.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Mở browser: http://localhost:5173

## Flow

### Upload Video
```
[User uploads video]
      ↓
[Send to /api/predict/video]
      ↓
[Face detection + preprocessing]
      ↓
[TS-CST Net inference]
      ↓
[Return HR result]
```

### Real-time
```
[Webcam captures frames]
      ↓
[Face detection (client-side)]
      ↓
[When face stable for 5s]
      ↓
[Send 128 frames to /api/predict/realtime]
      ↓
[Return HR result + PPG waveform]
```

