import logging, time
from discord_tron_client.classes.app_config import AppConfig
from datetime import datetime

class Auth:
    def __init__(self, config: AppConfig, access_token: str, refresh_token: str, expires_in: int, token_received_at: int):
        logging.info("Loaded auth ticket helper.")
        # Store your initial tokens and expiration time
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_in = expires_in  # Expiration time in seconds
        self.token_received_at = token_received_at
        self.base_url = config.get_master_url()

    def refresh_client_token(self, refresh_token):
        url = self.base_url + "/refresh_token"
        payload = {"refresh_token": refresh_token}
        import requests
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Error refreshing token: {}".format(response.text))

    def is_token_expired(self):
        token_received_at = datetime.fromisoformat(self.token_received_at).timestamp()
        expires_in = int(self.expires_in)
        test = time.time() >= (token_received_at + expires_in)
        logging.info(f"Token expired? {test}")
        return test