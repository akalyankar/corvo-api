from flask import Blueprint, jsonify
from datetime import datetime

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'corvo-api',
        'timestamp': datetime.utcnow().isoformat()
    })


@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check - verifies Corvo service is initialized"""
    try:
        from app.services.corvo_service import corvo_service
        return jsonify({
            'status': 'ready',
            'embedder_loaded': corvo_service._embedder is not None,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'error': str(e)
        }), 503

