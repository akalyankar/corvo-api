from flask import Flask
from flask_cors import CORS
from app.config import Config
from app.routes import health, normalization, categorization

def create_app():
    """Create and configure Flask application"""
    
    # Validate configuration
    Config.validate()
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app, origins=Config.CORS_ORIGINS)
    
    # Register blueprints
    app.register_blueprint(health.health_bp)
    app.register_blueprint(normalization.normalization_bp)
    app.register_blueprint(categorization.categorization_bp)
    
    # Initialize Corvo service on startup (Fixed - using app context instead of deprecated decorator)
    with app.app_context():
        try:
            from app.services.corvo_service import corvo_service
            print("✓ Corvo service initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Corvo service: {e}")
            raise
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Endpoint not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )

