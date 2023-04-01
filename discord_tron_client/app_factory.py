from flask import Flask
from discord_tron_client.config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Add any app configurations, blueprints, or extensions here

    return app