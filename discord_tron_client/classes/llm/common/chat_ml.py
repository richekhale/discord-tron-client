# A class for managing ChatML history for submission to LLM.
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.llm.common.token_tester import TokenTester

import json, logging

config = AppConfig()
app = AppConfig.flask

if app is None:
    raise Exception("Flask app is not initialized.")


class ChatML:
    def __init__(self, history: dict = None, token_limit: int = 2048):
        self.history = history or []
        self.user_config = config.get_user_config(self.user_id)
        # Pick up their current role from their profile.
        self.role = self.user_config["gpt_role"]
        self.reply = {}
        self.tokenizer = TokenTester()
        if token_limit > 2048:
            logging.warning(f"Token limit is too high. Setting to 2048.")
            token_limit = 2048
        self.token_limit = token_limit

    async def validate_reply(self):
        # If we are too long, maybe we can clean it up.
        logging.debug(f"Validating reply")
        if await self.is_reply_too_long():
            # let's clean up until it does fit.
            logging.debug(f"Eureka! We can enter Alzheimers mode.")
            await self.remove_history_until_reply_fits()
        return True

    async def get_reply_token_count(self):
        return self.tokenizer.get_token_count(json.dumps(self.reply))

    async def get_history_token_count(self):
        # Pad the value by 64 to accommodate for the metadata in the JSON we can't really count right here.
        return (
            self.tokenizer.get_token_count(json.dumps(await self.get_history())) + 512
        )

    # Format the history as a string for OpenAI.
    async def get_prompt(self):
        return json.dumps(await self.get_history())

    async def set_history(self, history: dict):
        self.history = history

    async def get_history(self):
        logging.debug(f"Conversation: {self.history}")
        return self.history

    async def is_history_empty(self):
        history = await self.get_history()
        if len(history) == 0:
            return True
        if history == []:
            return True
        return False

    def truncate_conversation_history(
        self, conversation_history, new_prompt, max_tokens=2048
    ):
        # Calculate tokens for new_prompt
        new_prompt_token_count = self.tokenizer.get_token_count(new_prompt)
        if new_prompt_token_count >= max_tokens:
            raise ValueError("The new prompt alone exceeds the maximum token limit.")

        # Calculate tokens for conversation_history
        conversation_history_token_counts = [
            len(self.tokenizer.tokenize(entry)) for entry in conversation_history
        ]
        total_tokens = sum(conversation_history_token_counts) + new_prompt_token_count

        # Truncate conversation history if total tokens exceed max_tokens
        while total_tokens > max_tokens:
            conversation_history.pop(0)  # Remove the oldest entry
            conversation_history_token_counts.pop(
                0
            )  # Remove the oldest entry's token count
            total_tokens = (
                sum(conversation_history_token_counts) + new_prompt_token_count
            )

        return conversation_history
