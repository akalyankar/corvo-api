#!/usr/bin/env python
"""Run the Flask API server"""
import sys

print("Starting Corvo API...", flush=True)
print("=" * 50, flush=True)

from app.main import create_app

if __name__ == '__main__':
    try:
        app = create_app()
        print("=" * 50, flush=True)
        print(f"Starting Flask server on http://0.0.0.0:8000", flush=True)
        print("Press CTRL+C to stop", flush=True)
        print("=" * 50, flush=True)
        app.run(debug=True, host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\nFailed to start server: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


