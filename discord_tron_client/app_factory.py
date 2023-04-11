from flask import Flask
from discord_tron_client.classes.app_config import AppConfig

def create_app(config_class=AppConfig):
    app = Flask(__name__)
    # app.config.from_object(config_class)

    # Add any app configurations, blueprints, or extensions here

    return app