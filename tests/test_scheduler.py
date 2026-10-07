import asyncio
from app.rtc import LatestFrameProcessor


def test_replace_keeps_latest():
    async def run():
        q = asyncio.Queue(maxsize=1)
        LatestFrameProcessor._replace(q, 1)
        LatestFrameProcessor._replace(q, 2)
        assert await q.get() == 2
    asyncio.run(run())
