"""
Detector module: YOLO-based exuvia detection
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

YOLO = None
YOLO_IMPORT_ERROR = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception as e:
    YOLO_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


class ExuviaDetector:
    """YOLO-based detector for exuvia counting"""
    
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initialize detector.
        
        Args:
            model_name: Model to load (e.g., "yolov8n.pt", "yolov8s.pt")
        """
        self.model = None
        self.model_name = model_name
        self.last_results = None

        if YOLO is None:
            logger.warning(
                "Ultralytics not available. Detection is disabled until optional ML deps are installed. "
                f"Import error: {YOLO_IMPORT_ERROR}"
            )
            return
        
        try:
            logger.info(f"Loading model: {model_name}")
            self.model = YOLO(model_name)
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def detect(self, image_source, confidence=0.5, iou=0.45, imgsz=None):
        """
        Run detection on image.
        
        Args:
            image_source: Path to image file or numpy array
            confidence: Confidence threshold (0-1)
            iou: IoU threshold for NMS
            imgsz: Optional YOLO inference image size
        
        Returns:
            dict with keys: 'count', 'boxes', 'confidences', 'image_with_boxes', 'raw_results'
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            predict_kwargs = {
                "source": image_source,
                "conf": confidence,
                "iou": iou,
                "verbose": False,
            }
            if imgsz is not None:
                predict_kwargs["imgsz"] = int(imgsz)

            results = self.model.predict(**predict_kwargs)
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # Extract data
                count = len(boxes)
                box_coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = boxes.conf.cpu().numpy()
                
                # Get annotated image
                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                output = {
                    'count': count,
                    'boxes': box_coords,
                    'confidences': confidences,
                    'image_with_boxes': annotated_frame,
                    'raw_results': result,
                    'mean_confidence': float(np.mean(confidences)) if count > 0 else 0.0
                }
                
                self.last_results = output
                logger.info(f"Detection complete: {count} exuvia found")
                return output
            
            else:
                logger.warning("No detections found")
                return {
                    'count': 0,
                    'boxes': np.array([]),
                    'confidences': np.array([]),
                    'image_with_boxes': None,
                    'raw_results': None,
                    'mean_confidence': 0.0
                }
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None

    @staticmethod
    def _draw_count_badge(image_rgb, count):
        """Draw semi-transparent tile count badge in the top-left corner."""
        badge_h, badge_w = 32, 70
        x0, y0 = 4, 4
        x1, y1 = x0 + badge_w, y0 + badge_h

        overlay = image_rgb.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, image_rgb, 0.55, 0, image_rgb)
        cv2.putText(
            image_rgb,
            f"{int(count)}",
            (x0 + 12, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_tile_boxes(image_rgb, result):
        """Draw YOLO boxes directly onto tile pixels for sharper stitched output."""
        if result is None or getattr(result, "boxes", None) is None:
            return image_rgb

        out = image_rgb.copy()
        boxes = result.boxes
        names = getattr(result, "names", None)

        if len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
        cls_ids = boxes.cls.cpu().numpy().astype(np.int32) if boxes.cls is not None else np.array([])

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(out.shape[1] - 1, int(x2))
            y2 = min(out.shape[0] - 1, int(y2))

            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)

            score = float(conf[i]) if i < len(conf) else 0.0
            if i < len(cls_ids) and isinstance(names, dict):
                label = f"{names.get(int(cls_ids[i]), 'obj')} {score:.2f}"
            else:
                label = f"obj {score:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty1 = max(0, y1 - th - 6)
            ty2 = y1
            tx2 = min(out.shape[1] - 1, x1 + tw + 6)
            cv2.rectangle(out, (x1, ty1), (tx2, ty2), (255, 180, 0), -1)
            cv2.putText(out, label, (x1 + 3, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        return out

    def detect_fixed_grid(
        self,
        image_source,
        rows=6,
        cols=8,
        confidence=0.25,
        iou=0.45,
        imgsz=640,
        draw_boxes=True,
        use_batch=True,
    ):
        """Run per-tile inference on a fixed rows x cols grid and stitch results.

        This mirrors a training-aligned pipeline where each tile is processed independently
        (no overlap/NMS merge) and then stitched back to full frame.
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None

        if int(rows) < 1 or int(cols) < 1:
            logger.error("rows and cols must be >= 1")
            return None

        # Accept either image path or in-memory numpy array.
        if isinstance(image_source, (str, Path)):
            img_bgr = cv2.imread(str(image_source))
            if img_bgr is None:
                logger.error(f"Unable to read image: {image_source}")
                return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            arr = np.asarray(image_source)
            if arr.ndim != 3 or arr.shape[2] != 3:
                logger.error("image_source ndarray must be HxWx3")
                return None
            img_rgb = arr.copy()

        rows = int(rows)
        cols = int(cols)
        h, w = img_rgb.shape[:2]
        tile_h = max(1, h // rows)
        tile_w = max(1, w // cols)

        stitched = np.zeros_like(img_rgb)
        total_count = 0
        conf_values = []
        tile_count = 0

        tiles = []
        slots = []
        for r in range(rows):
            y1 = r * tile_h
            y2 = (r + 1) * tile_h if r < rows - 1 else h
            for c in range(cols):
                x1 = c * tile_w
                x2 = (c + 1) * tile_w if c < cols - 1 else w
                tile = img_rgb[y1:y2, x1:x2].copy()
                if tile.size == 0:
                    continue
                tiles.append(tile)
                slots.append((x1, y1, x2, y2))

        batch_results = None
        if tiles and use_batch:
            try:
                batch_results = self.model.predict(
                    source=tiles,
                    conf=float(confidence),
                    iou=float(iou),
                    imgsz=int(imgsz),
                    verbose=False,
                )
            except Exception as exc:
                logger.warning("Batch inference failed, falling back to sequential mode: %s", exc)
                batch_results = None

        if batch_results is None or len(batch_results) != len(tiles):
            # Fallback path if batch API shape differs or runtime fails.
            batch_results = []
            for tile_idx, tile in enumerate(tiles):
                try:
                    res = self.model.predict(
                        source=tile,
                        conf=float(confidence),
                        iou=float(iou),
                        imgsz=int(imgsz),
                        verbose=False,
                    )
                    batch_results.append(res[0] if res and len(res) > 0 else None)
                except Exception as exc:
                    logger.warning("Tile %s inference failed, continuing: %s", tile_idx, exc)
                    batch_results.append(None)

        for result, (x1, y1, x2, y2), tile in zip(batch_results, slots, tiles):
            tile_det_count = 0
            tile_annotated = tile

            if result is not None:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    tile_det_count = int(len(boxes))
                    conf_values.extend([float(v) for v in boxes.conf.cpu().numpy()])

                if draw_boxes:
                    tile_annotated = self._draw_tile_boxes(tile, result)

            slot_h = y2 - y1
            slot_w = x2 - x1
            if tile_annotated.shape[0] != slot_h or tile_annotated.shape[1] != slot_w:
                tile_annotated = cv2.resize(
                    tile_annotated,
                    (slot_w, slot_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            self._draw_count_badge(tile_annotated, tile_det_count)
            stitched[y1:y2, x1:x2] = tile_annotated

            total_count += tile_det_count
            tile_count += 1

        mean_conf = float(np.mean(conf_values)) if conf_values else 0.0
        std_conf = float(np.std(conf_values)) if conf_values else 0.0

        # Draw tile grid lines on the stitched image.
        for r in range(1, rows):
            gy = r * tile_h
            cv2.line(stitched, (0, gy), (w, gy), (80, 80, 80), 1)
        for c in range(1, cols):
            gx = c * tile_w
            cv2.line(stitched, (gx, 0), (gx, h), (80, 80, 80), 1)

        # Append a total count banner below the stitched image.
        banner_h = max(50, h // 20)
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        banner_text = (
            f"Total: {total_count}  |  {tile_count} tiles ({rows}x{cols})"
            f"  |  mean conf {mean_conf:.2f}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.55, banner_h / 70.0)
        thickness = 2
        (tw, th), _ = cv2.getTextSize(banner_text, font, font_scale, thickness)
        tx = max(0, (w - tw) // 2)
        ty = (banner_h + th) // 2
        cv2.putText(banner, banner_text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        stitched = np.vstack([stitched, banner])

        output = {
            "count": int(total_count),
            "boxes": np.array([]),
            "confidences": np.array(conf_values, dtype=np.float32),
            "image_with_boxes": stitched,
            "raw_results": None,
            "mean_confidence": mean_conf,
            "confidence_std": std_conf,
            "tile_count": int(tile_count),
            "rows": rows,
            "cols": cols,
        }
        self.last_results = output
        logger.info(
            "Fixed-grid detection complete: %s exuvia from %s tiles (%sx%s)",
            total_count,
            tile_count,
            rows,
            cols,
        )
        return output

    @staticmethod
    def _global_nms_xyxy(boxes, scores, iou_thresh=0.45):
        """Simple numpy NMS over xyxy boxes, returns kept indices."""
        if boxes is None or len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(scores)[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou_vals = np.where(union > 0.0, inter / union, 0.0)

            inds = np.where(iou_vals <= iou_thresh)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    @staticmethod
    def _center_dedupe_xyxy(boxes, scores, distance_ratio=0.35):
        """Greedy center-based dedupe to remove near-identical boxes from adjacent tiles."""
        if boxes is None or len(boxes) == 0:
            return np.array([], dtype=np.int32)

        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        widths = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
        scales = np.sqrt(widths * heights)

        order = np.argsort(scores)[::-1]
        kept = []
        suppressed = np.zeros(len(boxes), dtype=bool)

        for i in order:
            if suppressed[i]:
                continue
            kept.append(i)
            dx = centers_x - centers_x[i]
            dy = centers_y - centers_y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            radius = distance_ratio * np.minimum(scales, scales[i])
            near = dist <= radius
            suppressed = suppressed | near
            suppressed[i] = False

        return np.array(kept, dtype=np.int32)

    def detect_tiled(
        self,
        image_source,
        tile_size=640,
        overlap=80,
        confidence=0.35,
        iou=0.45,
        edge_margin_px=8,
        min_box_area_px=24,
    ):
        """Run tiled detection, remap boxes to full image, then apply global NMS."""
        if self.model is None:
            logger.error("Model not loaded")
            return None

        img_bgr = cv2.imread(str(image_source))
        if img_bgr is None:
            logger.error(f"Unable to read image: {image_source}")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        stride = max(1, int(tile_size) - int(overlap))
        coords = set()

        for y in range(0, max(1, h - int(tile_size) + 1), stride):
            for x in range(0, max(1, w - int(tile_size) + 1), stride):
                if x + int(tile_size) <= w and y + int(tile_size) <= h:
                    coords.add((x, y))

        if w % stride != 0 and w >= int(tile_size):
            x = w - int(tile_size)
            for y in range(0, max(1, h - int(tile_size) + 1), stride):
                if y + int(tile_size) <= h:
                    coords.add((x, y))

        if h % stride != 0 and h >= int(tile_size):
            y = h - int(tile_size)
            for x in range(0, max(1, w - int(tile_size) + 1), stride):
                if x + int(tile_size) <= w:
                    coords.add((x, y))

        if w % stride != 0 and h % stride != 0 and w >= int(tile_size) and h >= int(tile_size):
            coords.add((w - int(tile_size), h - int(tile_size)))

        coords = sorted(coords, key=lambda c: (c[1], c[0]))

        all_boxes = []
        all_confs = []
        tile_count = 0

        for (x0, y0) in coords:
            tile = img_rgb[y0:y0 + int(tile_size), x0:x0 + int(tile_size)]
            if tile.size == 0:
                continue

            results = self.model.predict(source=tile, conf=confidence, iou=iou, verbose=False)
            if not results or len(results) == 0:
                tile_count += 1
                continue

            r = results[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                tile_count += 1
                continue

            local_xyxy = boxes.xyxy.cpu().numpy()
            local_conf = boxes.conf.cpu().numpy()

            for b, c in zip(local_xyxy, local_conf):
                gx1 = float(b[0] + x0)
                gy1 = float(b[1] + y0)
                gx2 = float(b[2] + x0)
                gy2 = float(b[3] + y0)
                all_boxes.append([gx1, gy1, gx2, gy2])
                all_confs.append(float(c))

            tile_count += 1

        if len(all_boxes) == 0:
            return {
                "count": 0,
                "boxes": np.array([]),
                "confidences": np.array([]),
                "image_with_boxes": img_rgb,
                "raw_results": None,
                "mean_confidence": 0.0,
                "confidence_std": 0.0,
                "tile_count": tile_count,
                "mode": "tiled_overlap_global_merge",
                "edge_margin_px": int(edge_margin_px),
                "min_box_area_px": float(min_box_area_px),
                "filtered_edge_count": 0,
                "filtered_tiny_count": 0,
            }

        boxes_np = np.array(all_boxes, dtype=np.float32)
        confs_np = np.array(all_confs, dtype=np.float32)
        keep_idx = self._global_nms_xyxy(boxes_np, confs_np, iou_thresh=float(iou))

        nms_boxes = boxes_np[keep_idx] if len(keep_idx) > 0 else np.empty((0, 4), dtype=np.float32)
        nms_confs = confs_np[keep_idx] if len(keep_idx) > 0 else np.array([], dtype=np.float32)

        dedupe_idx = self._center_dedupe_xyxy(nms_boxes, nms_confs, distance_ratio=0.35)
        final_boxes = nms_boxes[dedupe_idx] if len(dedupe_idx) > 0 else np.empty((0, 4), dtype=np.float32)
        final_confs = nms_confs[dedupe_idx] if len(dedupe_idx) > 0 else np.array([], dtype=np.float32)

        # Final post-merge cleanup for tray images:
        # 1) suppress boxes hugging frame edges, 2) suppress tiny texture noise.
        edge_margin_px = max(0, int(edge_margin_px))
        min_box_area_px = max(0.0, float(min_box_area_px))
        kept_boxes = []
        kept_confs = []
        filtered_edge_count = 0
        filtered_tiny_count = 0

        for b, c in zip(final_boxes, final_confs):
            x1, y1, x2, y2 = [float(v) for v in b]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area = bw * bh

            if area < min_box_area_px:
                filtered_tiny_count += 1
                continue

            touches_edge = (
                x1 <= edge_margin_px
                or y1 <= edge_margin_px
                or x2 >= float(w - 1 - edge_margin_px)
                or y2 >= float(h - 1 - edge_margin_px)
            )
            if touches_edge:
                filtered_edge_count += 1
                continue

            kept_boxes.append([x1, y1, x2, y2])
            kept_confs.append(float(c))

        final_boxes = np.array(kept_boxes, dtype=np.float32) if kept_boxes else np.empty((0, 4), dtype=np.float32)
        final_confs = np.array(kept_confs, dtype=np.float32) if kept_confs else np.array([], dtype=np.float32)

        annotated = img_rgb.copy()
        for b, c in zip(final_boxes, final_confs):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 50), 2)
            cv2.putText(
                annotated,
                f"{c:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (50, 220, 50),
                1,
                cv2.LINE_AA,
            )

        out = {
            "count": int(len(final_boxes)),
            "boxes": final_boxes,
            "confidences": final_confs,
            "image_with_boxes": annotated,
            "raw_results": None,
            "mean_confidence": float(np.mean(final_confs)) if len(final_confs) > 0 else 0.0,
            "confidence_std": float(np.std(final_confs)) if len(final_confs) > 0 else 0.0,
            "tile_count": tile_count,
            "mode": "tiled_overlap_global_merge",
            "edge_margin_px": int(edge_margin_px),
            "min_box_area_px": float(min_box_area_px),
            "filtered_edge_count": int(filtered_edge_count),
            "filtered_tiny_count": int(filtered_tiny_count),
        }
        self.last_results = out
        logger.info(f"Tiled detection complete: {out['count']} exuvia from {tile_count} tiles")
        return out
    
    def detect_and_save(self, image_path, output_dir=None, confidence=0.5, iou=0.45):
        """
        Detect and save annotated image.
        
        Args:
            image_path: Path to image
            output_dir: Where to save (default: data/detections)
            confidence: Confidence threshold
            iou: IoU threshold for NMS
        
        Returns:
            dict with detection results and saved path
        """
        if output_dir is None:
            output_dir = Path("data/detections")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = self.detect(str(image_path), confidence=confidence, iou=iou)
        
        if results and results['image_with_boxes'] is not None:
            # Save annotated image
            img_bgr = cv2.cvtColor(results['image_with_boxes'], cv2.COLOR_RGB2BGR)
            
            base_name = Path(image_path).stem
            output_path = output_dir / f"{base_name}_detected.jpg"
            cv2.imwrite(str(output_path), img_bgr)
            
            results['saved_path'] = str(output_path)
            logger.info(f"Annotated image saved: {output_path}")
        
        return results
    
    def is_available(self):
        """Check if model is loaded"""
        return self.model is not None
    
    @staticmethod
    def get_available_models():
        """List common Ultralytics model presets"""
        return {
            "nano": "yolo11n.pt",
            "small": "yolo11s.pt",
            "medium": "yolo11m.pt",
        }


# Singleton for Streamlit
_detector_instance = None

def get_detector(model_name="yolov8n.pt"):
    """Get or create detector"""
    global _detector_instance
    if _detector_instance is None or _detector_instance.model_name != model_name:
        _detector_instance = ExuviaDetector(model_name)
    return _detector_instance
