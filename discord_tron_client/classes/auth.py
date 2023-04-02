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

    # When it's expired, we have to refresh the token.
    def refresh_client_token(self, refresh_token):
        url = self.base_url + "/refresh_token"
        payload = {"refresh_token": refresh_token}
        import requests
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            self.write_auth_ticket(response)
            return response.json()
        else:
            raise Exception("Error refreshing token: {}".format(response.text))

    # Before the token expires, we can get a new one normally.
    def get_access_token(self):
        url = self.base_url + "/authorize"
        from discord_tron_client.classes.app_config import AppConfig
        config = AppConfig()
        api_key = config.get_master_api_key()
        auth_ticket = config.get_auth_ticket()
        payload = { "api_key": api_key, "client_id": auth_ticket["client_id"] }

        import requests
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            self.write_auth_ticket(response)
            new_ticket = response.json()['access_token']
            self.access_token = new_ticket["access_token"]
            self.expires_in = new_ticket["expires_in"]
            self.token_received_at = new_ticket["issued_at"]
            return response.json()
        else:
            raise Exception("Error refreshing token: {}".format(response.text))

    def write_auth_ticket(self, response):
        import json
        from discord_tron_client.classes.app_config import AppConfig
        config = AppConfig()
        with open(config.auth_ticket_path, "w") as f:
            f.write(json.dumps(response.json()))

    def is_token_expired(self):
        token_received_at = datetime.fromisoformat(self.token_received_at).timestamp()
        expires_in = int(self.expires_in)
        test = time.time() >= (token_received_at + expires_in)
        logging.info(f"Token expired? {test}")
        return test

    # Request an access token from the auth server, refreshing it if necessary.
    def get(self):
        try:
            is_expired = self.is_token_expired()
        except Exception as e:
            logging.error(f"Error checking token expiration: {e}")
            is_expired = True
        if is_expired:
            logging.warning("Access token is expired. Attempting to refresh...")
            current_ticket = self.refresh_client_token(current_ticket["refresh_token"])
            import json
            print(f"New ticket: {json.dumps(current_ticket, indent=4)}")
        self.get_access_token()