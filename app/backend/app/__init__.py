from flask import Flask
from flask_cors import CORS
from flasgger import Swagger
from .config import Config
from .routes import init_routes


def create_app() -> Flask:
    """
    Create and configure the Flask application instance.

    Initialises the Flask app, loads configuration, enables CORS,
    and registers all API routes via Flasgger + init_routes.

    Returns:
        Flask: A fully configured Flask application instance.
    """
    app = Flask(__name__)
    swagger = Swagger(app)
    app.config.from_object(Config)
    CORS(app)  # allow cross-origin requests
    init_routes(app)  # register API routes
    return app
