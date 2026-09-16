import asyncio
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

from full_stand.pipeline import PlatePipeline

QUEUED: str = 'queued'
RUNNING: str = 'running'
DONE: str = 'done'
FAILED: str = 'failed'
CANCELLED: str = 'cancelled'

TERMINAL: frozenset = frozenset((DONE, FAILED, CANCELLED))
SUBSCRIBER_BUFFER: int = 256


@dataclass(slots=True)
class Job:
    id: str
    filename: str
    size: int
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

    def summary(self, position: Optional[int] = None) -> Dict:
        plates: List[Dict] = (self.result or {}).get('plates', [])
        timing: Dict = (self.result or {}).get('timing_ms', {})
        return {
            'id': self.id,
            'filename': self.filename,
            'size': self.size,
            'status': self.status,
            'created': self.created,
            'started': self.started,
            'finished': self.finished,
            'position': position,
            'error': self.error,
            'plates': len(plates),
            'total_ms': timing.get('total'),
            'texts': [
                {
                    'text': plate['text'],
                    'subtype': plate['subtype'],
                    'confidence': plate['confidence'],
                    'valid': plate['valid'],
                    'readable': plate['readable'],
                }
                for plate in plates
            ],
        }


class JobQueue:
    def __init__(self, pipeline: PlatePipeline, history_limit: int, queue_limit: int) -> None:
        self._pipeline: PlatePipeline = pipeline
        self._history_limit: int = history_limit
        self._queue_limit: int = queue_limit
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._payloads: Dict[str, bytes] = {}
        self._pending: Deque[str] = deque()
        self._channel: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers: Set[asyncio.Queue] = set()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='plate-worker',
        )
        self._worker: Optional[asyncio.Task] = None
        self._processed: int = 0
        self._failed: int = 0
        self._busy: Optional[str] = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._loop(), name='plate-queue')

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        for subscriber in tuple(self._subscribers):
            subscriber.put_nowait(None)
        self._subscribers.clear()
        self._payloads.clear()

    def subscribe(self) -> asyncio.Queue:
        subscriber: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_BUFFER)
        self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: asyncio.Queue) -> None:
        self._subscribers.discard(subscriber)

    def _publish(self, event: Dict) -> None:
        for subscriber in tuple(self._subscribers):
            try:
                subscriber.put_nowait(event)
            except asyncio.QueueFull:
                self._subscribers.discard(subscriber)

    def _position(self, job_id: str) -> Optional[int]:
        try:
            return self._pending.index(job_id)
        except ValueError:
            return None

    def _emit(self, job: Job) -> None:
        self._publish({'type': 'job', 'job': job.summary(self._position(job.id))})

    def _emit_stats(self) -> None:
        self._publish({'type': 'stats', 'stats': self.stats()})

    def _evict(self) -> None:
        while len(self._jobs) > self._history_limit:
            for job_id, job in self._jobs.items():
                if job.status in TERMINAL:
                    del self._jobs[job_id]
                    break
            else:
                return

    def submit(self, filename: str, payload: bytes) -> Job:
        if len(self._pending) >= self._queue_limit:
            raise OverflowError(f'queue is full ({self._queue_limit} jobs)')
        job: Job = Job(id=uuid.uuid4().hex[:12], filename=filename, size=len(payload))
        self._jobs[job.id] = job
        self._payloads[job.id] = payload
        self._pending.append(job.id)
        self._channel.put_nowait(job.id)
        self._evict()
        self._emit(job)
        self._emit_stats()
        return job

    def cancel(self, job_id: str) -> bool:
        job: Optional[Job] = self._jobs.get(job_id)
        if job is None or job.status != QUEUED:
            return False
        job.status = CANCELLED
        job.finished = time.time()
        self._payloads.pop(job_id, None)
        try:
            self._pending.remove(job_id)
        except ValueError:
            pass
        self._emit(job)
        self._emit_stats()
        return True

    def clear(self) -> int:
        removed: int = 0
        for job_id in [k for k, v in self._jobs.items() if v.status in TERMINAL]:
            del self._jobs[job_id]
            removed += 1
        if removed:
            self._publish({'type': 'cleared'})
            self._emit_stats()
        return removed

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def describe(self, job: Job) -> Dict:
        return job.summary(self._position(job.id))

    def listing(self, limit: int = 200) -> List[Dict]:
        jobs: List[Job] = list(self._jobs.values())[-limit:]
        return [job.summary(self._position(job.id)) for job in reversed(jobs)]

    def stats(self) -> Dict:
        return {
            'queued': len(self._pending),
            'running': 1 if self._busy else 0,
            'processed': self._processed,
            'failed': self._failed,
            'history': len(self._jobs),
            'active': self._busy,
            'subscribers': len(self._subscribers),
        }

    async def _loop(self) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        while True:
            job_id: str = await self._channel.get()
            job: Optional[Job] = self._jobs.get(job_id)
            payload: Optional[bytes] = self._payloads.pop(job_id, None)
            if job is None or payload is None or job.status != QUEUED:
                continue

            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            job.status = RUNNING
            job.started = time.time()
            self._busy = job_id
            self._emit(job)
            self._emit_stats()

            try:
                job.result = await loop.run_in_executor(self._executor, self._pipeline, payload)
                job.status = DONE
                self._processed += 1
            except asyncio.CancelledError:
                job.status = CANCELLED
                job.finished = time.time()
                self._busy = None
                self._emit(job)
                raise
            except Exception as error:
                job.status = FAILED
                job.error = f'{type(error).__name__}: {error}'
                self._failed += 1
            job.finished = time.time()
            self._busy = None
            self._emit(job)
            self._emit_stats()
