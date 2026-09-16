import argparse
import asyncio
import io
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from full_stand.config import StandConfig
from full_stand.jobs import DONE, Job, JobQueue
from full_stand.pipeline import PlatePipeline

STATIC: Path = Path(__file__).resolve().parent / 'static'
HEARTBEAT_SECONDS: float = 15.0

config: StandConfig = StandConfig()
state: Dict[str, object] = {'pipeline': None, 'queue': None, 'warmup_ms': 0.0}


def pipeline() -> PlatePipeline:
    instance = state['pipeline']
    if instance is None:
        raise HTTPException(status_code=503, detail='pipeline is not loaded')
    return instance


def queue() -> JobQueue:
    instance = state['queue']
    if instance is None:
        raise HTTPException(status_code=503, detail='queue is not running')
    return instance


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    config.validate()
    engine: PlatePipeline = PlatePipeline(config)
    state['warmup_ms'] = engine.warmup()
    state['pipeline'] = engine
    jobs: JobQueue = JobQueue(engine, config.history_limit, config.queue_limit)
    jobs.start()
    state['queue'] = jobs
    print(f'stand ready on {engine.device} | detector {engine.detector_parameters:,} params | '
          f'ocr {engine.ocr_parameters:,} params | warmup {state["warmup_ms"]:.0f} ms', flush=True)
    yield
    await jobs.stop()
    state['queue'] = None
    state['pipeline'] = None


app: FastAPI = FastAPI(title='PlateStand', version='1.0.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')


@app.get('/', include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / 'index.html')


@app.get('/api/health')
async def health() -> Dict:
    engine: PlatePipeline = pipeline()
    return {
        'status': 'ok',
        'device': str(engine.device),
        'detector': {
            'checkpoint': str(config.detector),
            'parameters': engine.detector_parameters,
            'classes': engine.detector_classes,
            'imgsz': config.imgsz,
            'confidence': config.confidence,
            'iou': config.iou,
        },
        'recognizer': {
            'checkpoint': str(config.recognizer),
            'parameters': engine.ocr_parameters,
            'weights': 'ema' if config.use_ema else 'raw',
            'graphs': engine.predictor.captured,
            'traced': engine.predictor.traced,
            'base_pad': config.base_pad,
        },
        'warmup_ms': state['warmup_ms'],
        'limits': {
            'upload_bytes': config.max_upload_bytes,
            'queue': config.queue_limit,
            'history': config.history_limit,
            'max_detections': config.max_detections,
        },
        'stats': queue().stats(),
    }


@app.post('/api/jobs')
async def submit(files: List[UploadFile] = File(...)) -> Dict:
    jobs: JobQueue = queue()
    accepted: List[Dict] = []
    rejected: List[Dict] = []
    for upload in files:
        payload: bytes = await upload.read()
        name: str = upload.filename or 'upload'
        if not payload:
            rejected.append({'filename': name, 'reason': 'empty file'})
            continue
        if len(payload) > config.max_upload_bytes:
            limit: int = config.max_upload_bytes // (1024 * 1024)
            rejected.append({'filename': name, 'reason': f'exceeds {limit} MB'})
            continue
        try:
            accepted.append(jobs.describe(jobs.submit(name, payload)))
        except OverflowError as error:
            rejected.append({'filename': name, 'reason': str(error)})
    if not accepted and rejected:
        raise HTTPException(status_code=413, detail=rejected[0]['reason'])
    return {'accepted': accepted, 'rejected': rejected}


@app.get('/api/jobs')
async def listing(limit: int = Query(200, ge=1, le=1000)) -> Dict:
    jobs: JobQueue = queue()
    return {'jobs': jobs.listing(limit), 'stats': jobs.stats()}


@app.get('/api/jobs/{job_id}')
async def detail(job_id: str) -> Dict:
    job: Optional[Job] = queue().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='job not found')
    return {'job': job.summary(), 'result': job.result}


@app.delete('/api/jobs/{job_id}')
async def cancel(job_id: str) -> Dict:
    if not queue().cancel(job_id):
        raise HTTPException(status_code=409, detail='job is not cancellable')
    return {'cancelled': job_id}


@app.post('/api/jobs/clear')
async def clear() -> Dict:
    return {'removed': queue().clear()}


@app.get('/api/export.csv')
async def export() -> StreamingResponse:
    jobs: JobQueue = queue()
    buffer: io.StringIO = io.StringIO()
    buffer.write('image;plate_num;subtype;confidence;detection;valid\n')
    for summary in reversed(jobs.listing(config.history_limit)):
        if summary['status'] != DONE:
            continue
        job: Optional[Job] = jobs.get(summary['id'])
        if job is None or job.result is None:
            continue
        name: str = job.filename.replace(';', ',')
        for plate in job.result['plates']:
            buffer.write(f'{name};{plate["text"]};{plate["subtype"]};'
                         f'{plate["confidence"]:.4f};{plate["detection"]:.4f};'
                         f'{int(plate["valid"])}\n')
    headers: Dict[str, str] = {'Content-Disposition': 'attachment; filename="plates.csv"'}
    return StreamingResponse(iter([buffer.getvalue()]), media_type='text/csv', headers=headers)


@app.get('/api/stream')
async def stream() -> StreamingResponse:
    jobs: JobQueue = queue()
    subscriber: asyncio.Queue = jobs.subscribe()

    async def publish() -> AsyncIterator[str]:
        try:
            yield f'data: {json.dumps({"type": "stats", "stats": jobs.stats()})}\n\n'
            while True:
                try:
                    event = await asyncio.wait_for(subscriber.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
                    continue
                if event is None:
                    break
                yield f'data: {json.dumps(event)}\n\n'
        finally:
            jobs.unsubscribe(subscriber)

    headers: Dict[str, str] = {
        'Cache-Control': 'no-cache, no-transform',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive',
    }
    return StreamingResponse(publish(), media_type='text/event-stream', headers=headers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='full_stand', description='plate detection and reading stand')
    parser.add_argument('--host', default=config.host)
    parser.add_argument('--port', type=int, default=config.port)
    parser.add_argument('--detector', default=str(config.detector))
    parser.add_argument('--recognizer', default=str(config.recognizer))
    parser.add_argument('--device', default=None, help='cuda, cuda:1, cpu')
    parser.add_argument('--conf', type=float, default=config.confidence)
    parser.add_argument('--iou', type=float, default=config.iou)
    parser.add_argument('--imgsz', type=int, default=config.imgsz)
    parser.add_argument('--base-pad', type=float, default=config.base_pad)
    parser.add_argument('--max-det', type=int, default=config.max_detections)
    parser.add_argument('--raw', action='store_true', help='use raw weights instead of ema')
    parser.add_argument('--eager', action='store_true', help='disable cuda graph capture')
    parser.add_argument('--script', action='store_true',
                        help='torchscript tracing: ~8%% faster reads, ~6 min startup')
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    config.host = args.host
    config.port = args.port
    config.detector = Path(args.detector)
    config.recognizer = Path(args.recognizer)
    config.device = args.device
    config.confidence = args.conf
    config.iou = args.iou
    config.imgsz = args.imgsz
    config.base_pad = args.base_pad
    config.max_detections = args.max_det
    config.use_ema = not args.raw
    config.use_graphs = not args.eager
    config.use_script = args.script
    uvicorn.run(app, host=config.host, port=config.port, log_level='info')


if __name__ == '__main__':
    main()
