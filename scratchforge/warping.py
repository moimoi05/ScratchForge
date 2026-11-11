"""Native RGB stamp bending using a skeleton axis and inverse thin-plate spline."""
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _polyline(points, count):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, lengths > 1e-6]]
    if len(points) < 2:
        raise ValueError('Stroke centerline needs two distinct points')
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    positions = np.linspace(0, distance[-1], count)
    return np.column_stack([np.interp(positions, distance, points[:, axis]) for axis in range(2)])


def _farthest(start, neighbors):
    parents = {start: None}; queue = deque([start]); end = start
    while queue:
        end = queue.popleft()
        for other in neighbors[end]:
            if other not in parents:
                parents[other] = end; queue.append(other)
    return end, parents


def skeleton_axis(mask):
    """Analysis may be reduced; original RGB pixels are never resized here."""
    values = np.asarray(mask, dtype=np.uint8)
    if values.ndim != 2 or not np.isin(values, (0, 255)).all() or not values.any():
        raise ValueError('Stamp mask has no valid skeleton axis')
    ratio = min(1., 768/max(values.shape))
    if ratio < 1:
        values = cv2.resize(values, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    work = cv2.copyMakeBorder(values, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    skeleton = np.zeros_like(work); cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while work.any():
        eroded = cv2.erode(work, cross)
        skeleton |= cv2.subtract(work, cv2.dilate(eroded, cross))
        work = eroded
    coordinates = {tuple(point) for point in np.argwhere(skeleton > 0)}
    neighbors = {point: [(point[0]+dy, point[1]+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                         if (dy or dx) and (point[0]+dy, point[1]+dx) in coordinates]
                 for point in coordinates}
    remaining, longest = set(coordinates), []
    while remaining:
        endpoint, connected = _farthest(next(iter(remaining)), neighbors)
        far, parents = _farthest(endpoint, neighbors); path = []
        while far is not None:
            path.append(far); far = parents[far]
        if len(path) > len(longest):
            longest = path
        remaining.difference_update(connected)
    if len(longest) < 3:
        raise ValueError('Stamp skeleton axis is too short; choose an elongated scratch')
    axis = (np.asarray(longest, dtype=np.float64)[:, ::-1]-1)/ratio
    # Stable orientation avoids a different texture direction after every reload.
    if tuple(axis[0]) > tuple(axis[-1]):
        axis = axis[::-1]
    return axis


def _normals(points):
    tangent = np.gradient(points, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent/np.maximum(norm, 1e-6)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _kernel(left, right):
    squared = ((left[:, None, :]-right[None, :, :])**2).sum(axis=2)
    return squared*np.log(np.maximum(squared, 1e-12))


def _tps_inverse(destination, source, grid):
    center = destination.mean(axis=0); extent = max(np.ptp(destination, axis=0).max(), 1.)
    controls = (destination-center)/extent
    count = len(controls)
    affine = np.column_stack([np.ones(count), controls])
    system = np.block([[_kernel(controls, controls)+np.eye(count)*1e-6, affine],
                       [affine.T, np.zeros((3, 3))]])
    targets = np.vstack([source, np.zeros((3, source.shape[1]))])
    try:
        coefficients = np.linalg.solve(system, targets)
    except np.linalg.LinAlgError as error:
        raise ValueError('Stroke centerline cannot form a stable TPS warp') from error
    output = np.empty((len(grid), source.shape[1]), dtype=np.float32)
    for start in range(0, len(grid), 16384):
        points = (grid[start:start+16384]-center)/extent
        features = np.column_stack([_kernel(points, controls), np.ones(len(points)), points])
        output[start:start+len(points)] = features@coefficients
    return output


def _end_profile(seed, positions):
    rng = np.random.default_rng(seed)
    narrow, wide = rng.uniform(.16, .35), rng.uniform(.55, .78)
    short, long = rng.uniform(.16, .24), rng.uniform(.26, .36)
    reverse = bool(rng.integers(2))
    if reverse:
        narrow, wide = wide, narrow
        short, long = long, short
    start = np.clip(1 - positions / short, 0, 1)
    end = np.clip(1 - (1 - positions) / long, 0, 1)
    return 1 - (1 - narrow)*start - (1 - wide)*end, reverse


def _mix_end_colors(warped, rgb, mask, along, reverse):
    values = np.asarray(rgb)[np.asarray(mask) > 0]
    values = values[::max(1, len(values)//4096)]
    ordered = values[np.argsort(values @ np.array([.21, .72, .07]))]
    low, high = ordered[int(.2*(len(ordered)-1))], ordered[int(.8*(len(ordered)-1))]
    if reverse:
        low, high = high, low
    head = (.23*np.clip(1 - along/.25, 0, 1))[..., None]
    tail = (.23*np.clip(1 - (1 - along)/.25, 0, 1))[..., None]
    return np.rint(warped*(1-head-tail) + low*head + high*tail).clip(0, 255).astype('uint8')


def warp_stamp(rgb, mask, points, width, canvas_size, *, preserve_color=False, end_seed=None):
    if rgb.size != mask.size:
        raise ValueError('Stamp RGB and mask dimensions must match')
    if len(points) < 4 or len(points) % 2 or not np.isfinite(points).all() or not 1 <= width <= 512:
        raise ValueError('Invalid stroke centerline or width')
    vertices = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    target = _polyline(vertices, 18)
    source = _polyline(skeleton_axis(mask), 18)
    distance = cv2.distanceTransform(np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, 5)
    xs = np.clip(np.rint(source[:, 0]).astype(int), 0, mask.width-1)
    ys = np.clip(np.rint(source[:, 1]).astype(int), 0, mask.height-1)
    radius = np.maximum(distance[ys, xs], 1.)[:, None]
    source_side = _normals(source)*radius
    target_side = _normals(target)*width/2
    if end_seed is not None:
        positions = np.linspace(0, 1, len(target))
        widths, reverse = _end_profile(end_seed, positions)
        target_side *= widths[:, None]
    source_controls = np.concatenate([source-source_side, source, source+source_side])
    target_controls = np.concatenate([target-target_side, target, target+target_side])
    if end_seed is not None:
        source_controls = np.column_stack((source_controls, np.tile(positions, 3)))
    padding = width/2+3
    lo = np.maximum(0, np.floor(vertices.min(axis=0)-padding)).astype(int)
    hi = np.minimum(canvas_size, np.ceil(vertices.max(axis=0)+padding+1)).astype(int)
    if (hi <= lo).any():
        raise ValueError('Stamp brush lies outside the image')
    box = tuple(map(int, [*lo, *hi])); local_size = tuple(hi-lo)
    yy, xx = np.indices((local_size[1], local_size[0]), dtype=np.float32)
    grid = np.column_stack([(xx+lo[0]).ravel(), (yy+lo[1]).ravel()])
    mapping = _tps_inverse(target_controls, source_controls, grid).reshape(local_size[1], local_size[0], -1)
    pixels = np.asarray(rgb.convert('RGB'))
    if preserve_color:
        alpha = np.asarray(mask, dtype=np.float32) / 255
        pixels = np.dstack((pixels * alpha[..., None], alpha))
    warped = cv2.remap(pixels, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    if preserve_color:
        warped = np.rint(warped[..., :3] / np.maximum(warped[..., 3:], 1e-6)).clip(0, 255).astype('uint8')
    if end_seed is not None:
        warped = _mix_end_colors(warped, rgb, mask, np.clip(mapping[..., 2], 0, 1), reverse)
    binary = cv2.remap(np.asarray(mask), mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT)
    allowed = Image.new('L', local_size)
    ImageDraw.Draw(allowed).line([tuple(point-lo) for point in vertices], fill=255,
                                width=round(width), joint='curve')
    binary = np.where((binary >= 128) & (np.asarray(allowed) > 0), 255, 0).astype(np.uint8)
    if not binary.any():
        raise ValueError('Stamp brush produced no visible scratch; choose a different stamp or stroke')
    return Image.fromarray(warped), Image.fromarray(binary), box, {
        'warp': 'thin_plate_spline', 'vae_used': False, 'native_texture': True,
        'source_axis': source.tolist(), 'target_axis': target.tolist(),
        **({'end_profile_seed': end_seed, 'end_width_factors': [float(widths[0]), float(widths[-1])],
            'end_color_mix': 'source_palette'} if end_seed is not None else {})}
