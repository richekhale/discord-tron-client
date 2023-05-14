# TRON Client Worker 🤖

This project encompasses the "client" for the [TRON WebSocket hub.](https://github.com/bghira/discord-tron-master)

This codebase is capable of:

* 🖼 Generating Stable Diffusion images using ControlNet Tile Upscaler
* 🕵 Upscale images via ControlNet Tile v1.5
* ✔️ Run Stable Diffusion 2.1 images through 1.5 ControlNet pipeline
* 🔢 Use of Karras Sigmas for major quality improvement
* 🙊 OpenAI GPT integrations, including the option to enforce "jailbreaks"
* 📢 Bark TTS samples via [bghira/bghira-bark](https://github.com/bghira/bghira-bark)
* 📖 Language model predict/responses
* 👌 A focus on reliable and exceptional results, out of the box

This client code **requires** the TRON master node to be configured and
running properly, before it is useful. See that project page for details.

## Current State

This project is undergoing active development. It has no API for extensions, 
and currently, extending it for new functionality is an involved process.

Expect to have things break on update sometimes, especially through the
more experimental features. Older stuff is unlikely to break, and newer
things are more likely to have frequent changes.

## Requirements

This is developed on a laptop with 8GB of VRAM but that requires disabling
much of the project's abilities. You can use either Bark, or Stable Diffusion, 
but trying to run both in the bot on 8GB of VRAM, is not currently possible.

It's an "easy fix", but just not something that has been a focus.

For better luck, you want to have 24GB of VRAM. This allows use of native
1080p render outputs without VAE tiling, which greatly improves the coherence
of the resulting images.

For using this bot to run large language models, 24GB *can* work, but 48GB
is recommended. Again, it's an "easy fix" (the same one mentioned three
sentences ago) that just has yet to be implemented.

For the truly gluttenous and wasteful, an 80GB A100 is capable of loading
every aspect of this bot at once, and running all of those pipes concurrently.

## Installation

1. Create a python venv:

```bash
python -m venv .venv/
```

2. Enter the venv:

```bash
. .venv/bin/activate
```

3. Install poetry:

```bash
pip install poetry
```

4. Install all project requirements:

```bash
poetry install
```

## Configuring

1. Create an initial OAuth token and SSL key & cert via your master node.

2. Copy the resulting OAuth token output to `discord_tron_client/config/auth.json`

3. Copy the resulting SSL key and certificate files to `discord_tron_client/config`

4. Copy `discord_tron_client/config/example.json` to `discord_tron_client/config/config.json`

5. Update the values in `config.json` to point to your WebSocket server host and port:

```json
   "websocket_hub": {
        "host": "example.net",
        "port": 6789,
        "tls": true,
        "protocol": "wss"
   }
```

6. Run the client:

```bash
. .venv/bin/activate # Always ensure you're in the virtual environment first.
# If it says "poetry: command not found", you might need to rebuild your venv.
poetry run client > worker.log 2>&1
```

## Project Structure 🏗️

* `classes/`: A somewhat-structured folder for many useful classes.
  These classes handle the backend work for LLMs, image diffusion, etc.
* `config/`: You will have to set up the client, SSL keys & auth ticket here.
* `message/`: WebSocket message templates for sending requests to the master.
* `modules/`: Some of the WebSocket command handlers are located here.
* `ws_client/`: The WebSocket client code which handles auth and connection.
* `LICENSE`: The Silly Use License (SUL-1.0), because why not have some fun
  while coding? 😜

## Extending the Project 🚀

To add a new !command to the bot:

1. Add the !command cog processor and Job class to [the master project](https://github.com/bghira/discord-tron-master).
2. In this project, add a new entry to the `worker_processor` class, indicating
   your handler for an incoming payload in your module.
3. Add any relevant backend handlers to the `modules/` directory, following
   existing patterns implemented by other modules. Any improvement to this
   pattern is welcomed, as we're always looking to improve extensibility.
4. If the existing data matches an existing workflow for the user, eg,
   some text or an image to send - you can reuse the existing WebSocket
   command handlers on the master backend. Otherwise, you will have to
   extend that side to accept a new message type, or, do some kind of special
   handling.
F4. Test your changes extensively. No one wants to accept broken code.
5. Open a pull request, and hope for the best! 🤞

## Limitations 😬

### GPU memory exhaustion

1. Although this project is extensively tested on a laptop with 8GB of VRAM,
   currently, GPU memory can be easily exhausted if you're doing "lots of things".

   Furthermore, **8GB simply isn't enough for most of this bot to work correctly.**
   The TTS engine, language models, and image models, currently cannot signal
   to each other that they need to evacuate GPU memory space back to the CPU.

   **Workaround**: Restart the worker.
2. Due to the asynchronous nature of WebSockets, sending a message to the
   master node does not return a response. There's not any good infrastructure
   yet in the project to handle waiting for and receiving a given response to
   a given message. This means that "linear" programming style is hard to
   pull off in this project, and a routine that needs a lot of coordinated
   back-and-forth between subsystems is currently very difficult to do.

   **Example**: Using this project to tie the image generation, TTS, and LLMs
   into a replacement for the `bghira/chatgpt-video-generator` project would
   require implementing this infrastructure, opening modules up to this
   new workflow.
