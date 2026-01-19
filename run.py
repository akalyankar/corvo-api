#!/usr/bin/env python
"""Run the Flask API server"""
from app.main import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8000)


