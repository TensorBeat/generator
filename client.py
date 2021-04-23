import asyncio
from tensorbeat import sarosh_gen

from grpclib.client import Channel


async def main():
    channel = Channel(host="127.0.0.1", port=3491)
    service = sarosh_gen.SaroshGeneratorStub(channel)
    response = await service.generate_music(notes=["C4", "D4", "E4", "F4", "G4"])
    print(response)

    # don't forget to close the channel when done!
    channel.close()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
