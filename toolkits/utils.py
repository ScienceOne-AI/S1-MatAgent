import asyncio
def remove_quotes(path:str) -> str:
    return path.strip("'").strip('"')

async def yield_text(text, speed = 0.001):
    # simulate streaming output
    i = 0  # track the current character position
    while i < len(text):
        yield text[i]
        i += 1
        await asyncio.sleep(speed)  # simulate streaming return delay

async def yield_word(text, speed = 0.001):
    # simulate streaming output
    for word in text.split(" "):
        await asyncio.sleep(speed)
        yield word + " "