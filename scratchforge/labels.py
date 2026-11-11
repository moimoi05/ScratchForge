"""Binary scratch labels and COCO polygons in authoritative native coordinates."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from scratchforge.dataset import mask_for, valid_polygons

MAX_POLYGONS = 10_000
MAX_COORDINATES = 200_000


def _check_limits(polygons):
    if len(polygons) > MAX_POLYGONS or sum(map(len, polygons)) > MAX_COORDINATES:
        raise ValueError('Automatic labels exceed the polygon/coordinate limit; simplify or split the scene')


def _binary(mask):
    if isinstance(mask, Image.Image) and mask.mode == '1':
        mask = mask.convert('L')
    values = np.asarray(mask)
    if values.ndim != 2 or not np.isin(values, (0, 255)).all():
        raise ValueError('Use a binary mask (black=background, white=scratch)')
    return values.astype(np.uint8)


def _rectangles(values, offset, *, pixel_edges=False):
    """Disjoint run rectangles preserve holes and even one-pixel-wide scratches.

    COCO's polygon union cannot subtract a hole. Decompose those components into
    disjoint rectangles rather than silently filling holes or exporting fake RLE.
    Fractional right/bottom edges keep thin rectangles nondegenerate while PIL
    rasterization still selects exactly the original inclusive pixels.
    """
    active, rectangles = {}, []
    for y, row in enumerate(values):
        edges = np.diff(np.pad(row > 0, (1, 1)).astype(np.int8))
        spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        current = {tuple(map(int, span)) for span in spans}
        for span in active.keys() - current:
            rectangles.append((*span, active[span], y))
            if len(rectangles) > MAX_POLYGONS:
                raise ValueError('Mask exceeds the polygon limit; simplify or split the scene')
        active = {span: active.get(span, y) for span in current}
    if len(rectangles)+len(active) > MAX_POLYGONS:
        raise ValueError('Mask exceeds the polygon limit; simplify or split the scene')
    rectangles.extend((*span, start, len(values)) for span, start in active.items())
    ox, oy = offset
    inset = 0 if pixel_edges else .001
    return [[float(x+ox), float(y+oy), float(xx+ox-inset), float(y+oy),
             float(xx+ox-inset), float(yy+oy-inset), float(x+ox), float(yy+oy-inset)]
            for x, xx, y, yy in rectangles]


def mask_to_polygons(mask):
    values = _binary(mask)
    if not values.any():
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(values, connectivity=8)
    if count-1 > MAX_POLYGONS:
        raise ValueError('Mask exceeds the polygon limit; simplify or split the scene')
    result = []
    for label in range(1, count):
        x, y, width, height, _ = map(int, stats[label])
        local = np.where(labels[y:y+height, x:x+width] == label, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(local, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polygon = contours[0].reshape(-1, 2) if contours else np.empty((0, 2))
        if polygon.size > MAX_COORDINATES:
            raise ValueError('Mask contour exceeds the coordinate limit; simplify or split the scene')
        candidate = polygon.flatten().tolist()
        if len(contours) == 1 and len(polygon) >= 3 and abs(cv2.contourArea(polygon)) >= .5:
            rendered = Image.new('L', (width, height))
            ImageDraw.Draw(rendered).polygon(candidate, fill=255)
            if np.array_equal(local, np.asarray(rendered)):
                result.append((polygon + [x, y]).astype(float).flatten().tolist())
                continue
        result.extend(_rectangles(local, (x, y)))
    _check_limits(result)
    return result


def coco_mask(segmentation, size):
    """Decode pixel-edge COCO polygons using the standard rasterizer."""
    from pycocotools import mask as coco_tools

    _check_limits(segmentation)
    if not valid_polygons(segmentation, *size):
        raise ValueError('Invalid COCO polygon coordinates')
    encoded = coco_tools.frPyObjects(segmentation, size[1], size[0])
    return Image.fromarray(coco_tools.decode(coco_tools.merge(encoded)) * 255)


def coco_document(polygons, size, file_name='image.tif'):
    _check_limits(polygons)
    if polygons and not valid_polygons(polygons, *size):
        raise ValueError('Invalid COCO polygon coordinates')
    annotations = []
    segment_count = 0
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        left, top = np.maximum(0, np.floor(points.min(axis=0))).astype(int)
        right, bottom = np.minimum(size, np.floor(points.max(axis=0))+1).astype(int)
        if right <= left or bottom <= top:
            raise ValueError('Polygon does not contain any image pixels')
        local = (points-[left, top]).flatten().tolist()
        mask = mask_for([local], (right-left, bottom-top))
        box = mask.getbbox()
        if box is None:
            raise ValueError('Polygon does not contain any image pixels')
        segmentation = _rectangles(np.asarray(mask), (left, top), pixel_edges=True)
        segment_count += len(segmentation)
        if segment_count > MAX_POLYGONS or 8*segment_count > MAX_COORDINATES:
            raise ValueError('COCO labels exceed the polygon/coordinate limit; simplify or split the scene')
        annotations.append({'id': len(annotations)+1, 'image_id': 1, 'category_id': 1,
                            'segmentation': segmentation, 'iscrowd': 0,
                            'area': int(np.count_nonzero(mask)),
                            'bbox': [box[0]+int(left), box[1]+int(top), box[2]-box[0], box[3]-box[1]]})
    return {'images': [{'id': 1, 'file_name': file_name, 'width': size[0], 'height': size[1]}],
            'categories': [{'id': 1, 'name': 'scratch'}], 'annotations': annotations}


def difference_mask(before, after, support, threshold=8, min_area=3):
    if before.size != after.size or before.size != support.size:
        raise ValueError('Difference mask dimensions must match both RGB images')
    if not 1 <= threshold <= 255 or not 1 <= min_area <= before.width*before.height:
        raise ValueError('Invalid mask difference threshold or minimum component area')
    allowed = _binary(support) > 0
    left, right = np.asarray(before.convert('RGB'), dtype=np.int16), np.asarray(after.convert('RGB'), dtype=np.int16)
    changed = ((np.abs(right-left).max(axis=2) >= threshold) & allowed).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_area)*255
    return Image.fromarray(keep[labels])
