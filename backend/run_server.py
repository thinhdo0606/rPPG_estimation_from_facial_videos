"""
Quick start script for Heart Rate Estimation Web Backend
Run: python run_server.py
"""
import os
import sys
import socket

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def main():
    local_ip = get_local_ip()
    port = 8000
    
    print()
    print("=" * 60)
    print("  HEART RATE ESTIMATION - WEB API SERVER")
    print("=" * 60)
    print()
    print(f"  Local IP: {local_ip}")
    print(f"  Port: {port}")
    print()
    print("  Access URLs:")
    print(f"    API:     http://localhost:{port}")
    print(f"    Docs:    http://localhost:{port}/docs")
    print(f"    Network: http://{local_ip}:{port}")
    print()
    print("  Frontend should connect to:")
    print(f"    http://localhost:{port}")
    print()
    print("=" * 60)
    print("  Starting server... (Press Ctrl+C to stop)")
    print("=" * 60)
    print()
    
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main()

