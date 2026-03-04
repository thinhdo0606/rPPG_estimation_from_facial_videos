# Huong Dan Cai Dat Web App

## Yeu cau he thong

### Backend
- Python 3.9+
- pip

### Frontend
- Node.js 18+
- npm hoac yarn

---

## Buoc 1: Cai dat Backend

### 1.1 Tao virtual environment

```bash
cd web_app/backend

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Cai dat dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Copy model

Copy file `ts_cst_net_final.pth` vao:
```
web_app/backend/models/ts_cst_net_final.pth
```

### 1.4 Chay server

```bash
python run_server.py
```

Server se chay tai: http://localhost:8000

---

## Buoc 2: Cai dat Frontend

### 2.1 Cai dat dependencies

```bash
cd web_app/frontend
npm install
```

### 2.2 Chay development server

```bash
npm run dev
```

Frontend se chay tai: http://localhost:5173

---

## Buoc 3: Su dung Web App

### Real-time Mode
1. Mo http://localhost:5173/realtime
2. Cho phep truy cap webcam
3. Dat mat trong vong oval
4. Bam "Start Measurement"
5. Giu yen trong 5 giay countdown
6. Xem ket qua nhip tim

### Upload Mode
1. Mo http://localhost:5173/upload
2. Keo tha video hoac click de chon file
3. Bam "Analyze Heart Rate"
4. Doi xu ly va xem ket qua

---

## Cau truc thu muc

```
web_app/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI endpoints
│   │   ├── model_service.py  # ML model
│   │   ├── preprocessing.py  # Video processing
│   │   └── config.py         # Configuration
│   ├── models/               # Trained models (.pth)
│   ├── requirements.txt
│   └── run_server.py
│
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API calls
│   │   ├── hooks/           # Custom hooks
│   │   └── styles/          # CSS
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
└── docs/
    └── SETUP_GUIDE.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| POST | /api/predict/video | Upload video analysis |
| POST | /api/predict/realtime | Real-time frames analysis |
| WS | /ws/realtime/{id} | WebSocket for streaming |

---

## Troubleshooting

### Backend khong chay
- Kiem tra Python version: `python --version`
- Kiem tra virtual environment da active
- Kiem tra model file ton tai

### Frontend khong connect duoc Backend
- Kiem tra backend dang chay tai port 8000
- Kiem tra CORS trong config.py
- Kiem tra proxy trong vite.config.js

### Webcam khong hoat dong
- Cho phep quyen camera trong browser
- Thu dung Chrome/Firefox
- Kiem tra webcam co bi app khac su dung

### Ket qua khong chinh xac
- Dam bao anh sang tot
- Giu mat yen trong qua trinh do
- Dat mat dung trong vong oval

---

## Build cho Production

### Backend
```bash
# Chay production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
# Build static files
npm run build

# Preview build
npm run preview
```

---

## Lien he

Neu gap van de, vui long tao Issue tren GitHub.

