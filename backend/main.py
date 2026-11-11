"""Local ScratchForge API. Run one Uvicorn worker to serialize GPU access."""
import hashlib
import io
import re
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request as HttpRequest, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.images import MAX_TIFF_UPLOAD, preview_png, validate_tiff
from backend.jobs import Jobs
from backend.models import Annotations, Approval, Brush, ModelDownloadRequest, Placement, Request
from backend.segment import historical_stamp_asset, segment_router
from backend.store import Store, read_json
from scratchforge.dataset import ROOT, sha
from scratchforge.generation import generate_scene

ALLOWED_ORIGINS = {'http://localhost:5173', 'http://127.0.0.1:5173',
                   'http://localhost:8000', 'http://127.0.0.1:8000'}


def create_app(root=ROOT, infer=None, harmonize=None, downloads=None):
    store = Store(root); jobs = Jobs(store, infer, harmonize)
    if downloads is None:
        from backend.model_downloads import ModelDownloads
        downloads = ModelDownloads(root)

    @asynccontextmanager
    async def lifespan(app):
        yield
        downloads.close()
        jobs.close()

    application = FastAPI(title='ScratchForge F971 Studio', lifespan=lifespan)
    application.state.store = store; application.state.jobs = jobs
    application.add_middleware(CORSMiddleware, allow_origins=sorted(ALLOWED_ORIGINS),
                               allow_methods=['GET', 'POST', 'PUT'], allow_headers=['Content-Type'])

    @application.middleware('http')
    async def local_boundary(request: HttpRequest, call_next):
        origin = request.headers.get('origin')
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse({'detail': 'Untrusted request origin'}, status_code=403)
        content_length = request.headers.get('content-length')
        limit = 252 if request.url.path == '/api/library/import' else (
            MAX_TIFF_UPLOAD // (1024*1024) + 2 if request.url.path == '/api/upload' else 42)
        if content_length and (not content_length.isdigit() or int(content_length) > limit*1024*1024):
            return JSONResponse({'detail': f'Request exceeds {limit} MiB'}, status_code=413)
        return await call_next(request)

    @application.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({'detail': str(error)}, status_code=422)

    @application.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse({'detail': 'Requested record or asset was not found'}, status_code=404)

    @application.exception_handler(PermissionError)
    async def forbidden(request, error):
        return JSONResponse({'detail': str(error)}, status_code=409)

    @application.get('/api/health')
    def health():
        return {'engine': 'separate stamp harmonization and brush generation', 'trained': bool(jobs.training.checkpoints()),
                'ai': jobs.ai.metadata(), 'brush_models': {'sdxl': jobs.ai.metadata(), 'sd15': jobs.sd15.metadata()},
                'stamp_methods': ['classical_mixed', 'classical_laplacian', 'pctnet', 'harmonizer', 'rainnet']}

    @application.get('/api/models/status')
    def model_status():
        return {'models': downloads.status()}

    @application.post('/api/models/download')
    def model_download(request: ModelDownloadRequest):
        return downloads.start(request.model_id)

    @application.post('/api/models/cancel')
    def model_cancel(request: ModelDownloadRequest):
        return downloads.cancel(request.model_id)

    @application.get('/api/library')
    def get_library():
        return [stamp for stamp in store.library() if stamp.get('split') == 'train']

    @application.get('/api/dataset')
    def dataset():
        return read_json(store.root/'reports/verification.json', {'ready': False, 'message': 'Run scripts/import_f971.py and verify_dataset.py'})

    @application.get('/api/training')
    def training():
        checkpoints = jobs.training.checkpoints()
        return {'status': 'local LoRA training', 'checkpoints': checkpoints, 'fine_tuned_f971': bool(checkpoints)}

    from backend.training import GenerateStampsRequest, TrainLoraRequest

    @application.post('/api/training/lora')
    def train_lora(request: TrainLoraRequest):
        return jobs.submit_operation('train', request.model_dump())

    @application.post('/api/library/generate')
    def generate_stamps(request: GenerateStampsRequest):
        return jobs.submit_operation('generate_stamps', request.model_dump())

    @application.get('/api/backgrounds')
    def backgrounds():
        return [{'id': row['id'], 'width': row['width'], 'height': row['height'],
                 'url': f"/api/backgrounds/{row['id']}/image", 'thumbnail': f"/api/backgrounds/{row['id']}/thumbnail",
                 'source_image': Path(row['path']).name, 'split': row['split']}
                for row in store.index() if row['status'] == 'OK' and row['split'] == 'train']

    def background(identifier):
        return next((row for row in store.index() if str(row['id']) == identifier
                     and row['status'] == 'OK' and row['split'] == 'train'), None)

    def background_preview(identifier, thumbnail=False):
        row = background(identifier)
        if row is None:
            raise HTTPException(404, 'Background not found')
        key = hashlib.sha256(f"{row['id']}/{row.get('sha256')}".encode()).hexdigest()
        size = (480, 320) if thumbnail else None
        cached = store.out/f"background-{'thumb' if thumbnail else 'full'}-{key}.png"
        return FileResponse(preview_png(store.asset(row['path']), cached, size), media_type='image/png')

    @application.get('/api/backgrounds/{identifier}/image')
    def background_image(identifier: str):
        return background_preview(identifier)

    @application.get('/api/backgrounds/{identifier}/thumbnail')
    def background_thumbnail(identifier: str):
        return background_preview(identifier, thumbnail=True)

    @application.post('/api/backgrounds/{identifier}/open')
    def open_background(identifier: str):
        row = background(identifier)
        if row is None:
            raise HTTPException(404, 'Background not found')
        source_path = store.asset(row['path'])
        if not row.get('sha256') or sha(source_path) != row['sha256']:
            raise ValueError('Imported background changed after verification; rerun the data importer')
        with Image.open(source_path) as image:
            if image.size != (row['width'], row['height']) or image.mode != 'RGB':
                raise ValueError('Imported background dimensions or RGB mode changed')
            if source_path.suffix.lower() in {'.tif', '.tiff'}:
                validate_tiff(image)
            return store.register(image.convert('RGB'), row)

    @application.post('/api/upload')
    def upload(file: UploadFile):
        if Path(file.filename or '').suffix.lower() not in {'.tif', '.tiff'}:
            raise HTTPException(400, 'Use a .tif or .tiff image')
        content = file.file.read(MAX_TIFF_UPLOAD + 1)
        if len(content) > MAX_TIFF_UPLOAD:
            raise HTTPException(413, f'Max TIFF upload {MAX_TIFF_UPLOAD // (1024*1024)} MiB')
        try:
            with Image.open(io.BytesIO(content)) as image:
                validate_tiff(image)
                rgb = image.convert('RGB')
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(400, 'Invalid TIFF image or image too large') from error
        return store.register(rgb)

    @application.post('/api/jobs')
    def submit(request: Request):
        with store.lock:
            return jobs.submit(request.model_dump())

    @application.get('/api/jobs/{identifier}')
    def job(identifier: str):
        record = store.load('job', identifier)
        if record.get('result_id'):
            return {**record, 'result': store.load('result', record['result_id'])}
        return record

    @application.post('/api/jobs/{identifier}/cancel')
    def cancel_job(identifier: str):
        return jobs.cancel(identifier)

    @application.get('/api/results/{identifier}')
    def result(identifier: str):
        return store.load('result', identifier)

    @application.put('/api/results/{identifier}/annotations')
    def annotations(identifier: str, request: Annotations):
        return jobs.annotations(identifier, request.polygons)

    @application.post('/api/results/{identifier}/approve')
    def approve(identifier: str, request: Approval):
        return jobs.approve(identifier)

    @application.post('/api/results/{identifier}/export')
    def export(identifier: str):
        return jobs.export(identifier)

    @application.get('/api/exports')
    def exports():
        with store.lock:
            return [read_json(path) for path in sorted(store.out.glob('*-result.json'), key=lambda path: path.stat().st_mtime, reverse=True)]

    @application.get('/files/{name}')
    def files(name: str):
        if not re.fullmatch(r'[a-f0-9]{32}(?:(?:-input|-guidance|-edit|-labels)?\.png|\.tif|-polygon\.json|\.zip)', name):
            raise HTTPException(404, 'File not found')
        path = store.out/name
        if path.suffix == '.png' and re.fullmatch(r'[a-f0-9]{32}(?:-input)?', path.stem):
            native = path.with_suffix('.tif')
            if native.is_file():
                return FileResponse(preview_png(native, path), media_type='image/png')
        if not path.is_file():
            raise HTTPException(404, 'File not found')
        if path.suffix == '.tif' or name.endswith(('-labels.png', '-polygon.json')):
            identifier = name[:32]
            record = store.load('result', identifier)
            jobs.verify_image(record)
            jobs.verify_labels(record)
        if path.suffix == '.zip':
            record = store.load('result', path.stem)
            if record['review_status'] != 'approved' or not record.get('export_url'):
                raise HTTPException(409, 'Export approval is no longer valid')
            jobs.verify_image(record)
            jobs.verify_labels(record)
            jobs.verify_export_sources(record)
        return FileResponse(path)

    @application.get('/library-files/{name:path}')
    def library_file(name: str):
        from backend.library import registered_records
        allowed = {stamp.get(key) for stamp in registered_records(store)
                   for key in ('rgb', 'mask', 'thumbnail', 'preview_alpha', 'polygon', 'metadata')}
        if name not in allowed and not historical_stamp_asset(store, name):
            raise HTTPException(404, 'Library asset not found')
        return FileResponse(store.asset('data/library/' + name))

    application.include_router(segment_router(store))
    from backend.library_api import library_router
    application.include_router(library_router(store, jobs))
    dist = store.root/'frontend/dist'
    if dist.is_dir():
        application.mount('/', StaticFiles(directory=dist, html=True), name='studio')
    return application


app = create_app()
OUT = ROOT/'outputs'


def library():
    return app.state.store.library()


def render(image, request):
    """Compatibility helper for baseline callers; exports always require review."""
    rgb, mask, _, _ = generate_scene(image, {**request.model_dump(), 'engine': 'baseline', 'brush_method': 'procedural'}, library(), ROOT/'data/library')
    return rgb, mask, []
