"""Local ScratchForge API. Run one Uvicorn worker to serialize GPU access."""
import io
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request as HttpRequest, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.jobs import Jobs
from backend.models import Annotations, Approval, Brush, Placement, Request
from backend.store import Store, read_json
from scratchforge.dataset import ROOT, sha
from scratchforge.generation import generate_scene

ALLOWED_ORIGINS = {'http://localhost:5173', 'http://127.0.0.1:5173',
                   'http://localhost:8000', 'http://127.0.0.1:8000'}


def create_app(root=ROOT, infer=None):
    store = Store(root); jobs = Jobs(store, infer)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.executor.shutdown(wait=True, cancel_futures=False)

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
        if content_length and (not content_length.isdigit() or int(content_length) > 42*1024*1024):
            return JSONResponse({'detail': 'Request exceeds 42 MiB'}, status_code=413)
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
        return {'engine': 'baseline + local SDXL inpainting', 'trained': False, 'ai': jobs.ai.metadata()}

    @application.get('/api/library')
    def get_library():
        return [stamp for stamp in store.library() if stamp.get('split') == 'train']

    @application.get('/api/dataset')
    def dataset():
        return read_json(store.root/'reports/verification.json', {'ready': False, 'message': 'Run scripts/import_f971.py and verify_dataset.py'})

    @application.get('/api/training')
    def training():
        return {'status': 'pretrained inference only', 'checkpoints': [], 'fine_tuned_f971': False}

    @application.get('/api/backgrounds')
    def backgrounds():
        return [{'id': row['id'], 'width': row['width'], 'height': row['height'],
                 'url': f"/api/backgrounds/{row['id']}/image", 'thumbnail': f"/api/backgrounds/{row['id']}/thumbnail",
                 'source_image': row['source_image'], 'split': row['split']}
                for row in store.index() if row['status'] == 'OK' and row['split'] == 'train']

    def background(identifier):
        return next((row for row in store.index() if str(row['id']) == identifier
                     and row['status'] == 'OK' and row['split'] == 'train'), None)

    @application.get('/api/backgrounds/{identifier}/image')
    def background_image(identifier: str):
        row = background(identifier)
        if row is None:
            raise HTTPException(404, 'Background not found')
        return FileResponse(store.asset(row['path']))

    @application.get('/api/backgrounds/{identifier}/thumbnail')
    def background_thumbnail(identifier: str):
        row = background(identifier)
        if row is None:
            raise HTTPException(404, 'Background not found')
        import hashlib
        cached = store.out/('thumb-' + hashlib.sha256(str(row['id']).encode()).hexdigest() + '.jpg')
        if not cached.exists():
            with store.lock, Image.open(store.asset(row['path'])) as image:
                thumbnail = image.convert('RGB'); thumbnail.thumbnail((480, 320))
                thumbnail.save(cached, quality=80)
        return FileResponse(cached)

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
            return store.register(image.convert('RGB'), row)

    @application.post('/api/upload')
    async def upload(file: UploadFile):
        content = await file.read(40*1024*1024+1)
        if len(content) > 40*1024*1024:
            raise HTTPException(413, 'Max upload 40 MiB')
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.mode != 'RGB' or image.width*image.height > 50_000_000:
                    raise ValueError('Use an RGB image with at most 50 megapixels')
                return store.register(image)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(400, 'Invalid RGB image, unsupported format or image too large')

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
        if not re.fullmatch(r'[a-f0-9]{32}(?:-input|-guidance|-edit|-labels)?\.(?:png|zip)', name):
            raise HTTPException(404, 'File not found')
        path = store.out/name
        if not path.is_file():
            raise HTTPException(404, 'File not found')
        if path.suffix == '.zip':
            record = store.load('result', path.stem)
            if record['review_status'] != 'approved' or not record.get('export_url'):
                raise HTTPException(409, 'Export approval is no longer valid')
            jobs.verify_image(record)
            jobs.verify_export_sources(record)
        return FileResponse(path)

    @application.get('/library-files/{name}')
    def library_file(name: str):
        allowed = {stamp.get(key) for stamp in store.library() for key in ('rgb', 'mask', 'thumbnail')}
        if name not in allowed:
            raise HTTPException(404, 'Library asset not found')
        return FileResponse(store.asset('data/library/' + name))

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
    rgb, mask, _, _ = generate_scene(image, {**request.model_dump(), 'engine': 'baseline'}, library(), ROOT/'data/library')
    return rgb, mask, []
