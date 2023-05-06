import logging, time
from discord_tron_client.classes.app_config import AppConfig
from datetime import datetime
from threading import Semaphore

auth_semaphore = Semaphore(1)


class Auth:
    def __init__(
        self,
        config: AppConfig,
        access_token: str,
        refresh_token: str,
        expires_in: int,
        token_received_at: int,
    ):
        logging.info("Loaded auth ticket helper.")
        # Store your initial tokens and expiration time
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_in = expires_in  # Expiration time in seconds
        self.token_received_at = token_received_at
        self.base_url = config.get_master_url()
        self.config = config

    # When it's expired, we have to refresh the token.
    def refresh_client_token(self, refresh_token):
        logging.debug(f"Running refresh_client_token unconditionally.")
        url = self.base_url + "/refresh_token"
        payload = {"refresh_token": refresh_token}
        import requests

        response = requests.post(
            url, json=payload, verify=self.config.verify_master_ssl()
        )

        if response.status_code == 200:
            self.write_auth_ticket(response.json())
            return response.json()
        else:
            raise Exception("Error refreshing client token: {}".format(response.text))

    # Before the token expires, we can get a new one normally.
    def get_access_token(self):
        logging.debug(f"Running get_access_token unconditionally.")
        with auth_semaphore:
            logging.debug(f"Semaphore was free! Continuing to check token.")
            url = self.base_url + "/authorize"
            from discord_tron_client.classes.app_config import AppConfig

            config = AppConfig()
            api_key = config.get_master_api_key()
            auth_ticket = config.get_auth_ticket()
            if not auth_ticket:
                raise Exception("No auth ticket found?")
            payload = {"api_key": api_key, "client_id": auth_ticket["client_id"]}
            logging.debug(f"get_access_token payload: {payload}")

            import requests

            response = requests.post(
                url, json=payload, verify=self.config.verify_master_ssl()
            )
            print(f"Response: {response.text}")
            if response.status_code == 200:
                logging.debug(f"Received new auth ticket. Updating local copy.")
                new_ticket = response.json()["access_token"]
                self.write_auth_ticket(response.json()["access_token"])
                self.access_token = new_ticket["access_token"]
                self.expires_in = new_ticket["expires_in"]
                self.token_received_at = new_ticket["issued_at"]
                return response.json()
            else:
                raise Exception("Error refreshing token: {}".format(response.text))
        logging.debug(f"get_access_token is complete. returned Semaphore.")

    def write_auth_ticket(self, response):
        import json
        from discord_tron_client.classes.app_config import AppConfig

        config = AppConfig()
        with open(config.auth_ticket_path, "w") as f:
            if response is None:
                raise Exception("Error writing auth ticket: response is None")

            output = json.dumps(response)
            f.write(output)

    def is_token_expired(self):
        token_received_at = datetime.fromisoformat(self.token_received_at).timestamp()
        expires_in = int(self.expires_in) / 2
        test = time.time() >= (token_received_at + expires_in)
        logging.debug(
            f"Token received at {token_received_at} and expires {self.expires_in} seconds after, which we reduce to {expires_in} seconds after that. Current time is {time.time()}.. The result of the test is {test}"
        )
        if test:
            logging.warning(
                f"Token expired. Token received at {token_received_at}, expires in {expires_in}, current time is {time.time()}."
            )
        return test

    # Request an access token from the auth server, refreshing it if necessary.
    def get(self):
        attempts = 10
        for i in range(attempts):
            try:
                current_ticket = self.config.get_auth_ticket()
                is_expired = self.is_token_expired()
                if not is_expired:
                    return current_ticket

                live_token = self.get_access_token()["access_token"]
                logging.debug(f"Using existing token to refresh: {live_token}")
                return live_token
            except Exception as e:
                logging.error(f"Error checking token expiration: {e}")
                is_expired = True
            if is_expired:
                logging.warning("Access token is expired. Attempting to refresh...")
                current_ticket = self.refresh_client_token(
                    current_ticket["refresh_token"]
                )
                import json

                print(f"New ticket: {json.dumps(current_ticket, indent=4)}")
            return current_ticket
        raise Exception(
            "Unable to get access token after {} attempts.".format(attempts)
        )
