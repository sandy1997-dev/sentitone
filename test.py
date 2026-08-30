import websockets, asyncio, json, base64

async def test():
    async with websockets.connect(
        'ws://localhost:8080',
        additional_headers={"X-API-Key": "my_secret_key"}  # <-- FIXED
    ) as ws:
        await ws.recv()  # ready message
        await ws.send(json.dumps({
            "type": "synthesize",
            "text": "ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବା ପାଇଁ ମୁଁ ଏଠାରେ ଅଛି",
            "lang": "or-IN",
            "style": "default"
        }))
        while True:
            resp = json.loads(await ws.recv())
            if resp["type"] == "audio":
                with open("test_neural.wav", "wb") as f:
                    f.write(base64.b64decode(resp["audio_b64"]))
                print("Saved test_neural.wav")
                break
            elif resp["type"] == "error":
                print("Error:", resp["message"])
                break

asyncio.run(test())