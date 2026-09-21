"""
Camera module: Handles Picamera2 (Pi) and fallback webcam (desktop testing)
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import os
import time
import tempfile
import subprocess
import shutil

logger = logging.getLogger(__name__)

PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    logger.warning("Picamera2 not available. Fallback to OpenCV webcam.")


class CameraManager:
    """Unified camera interface for Pi HQ camera or USB webcam"""
    
    def __init__(self, use_pi_camera=True):
        """
        Initialize camera.
        
        Args:
            use_pi_camera: If True, try Pi camera; else use USB webcam
        """
        self.picam2 = None
        self.cap = None
        self.current_zoom = 1
        self.last_frame = None
        self.camera_type = None
        self.camera_resolution = None
        self.last_error = None
        self.rpicam_command = None
        self.rpicam_preview_command = None
        self.preview_process = None
        self.preview_log_handle = None
        self.preview_dir = Path("data/preview")
        self.preview_path = self.preview_dir / "latest.jpg"
        self.preview_config = None
        self._last_preview_jpeg = None
        
        if use_pi_camera:
            self._init_pi_camera()
        else:
            self._init_usb_camera()
    
    def _init_pi_camera(self):
        """Initialize Raspberry Pi camera, preferring the working CLI path."""
        self._init_rpicam_cli()
        if self.camera_type is not None:
            return

        if not PICAMERA2_AVAILABLE:
            self.last_error = "Picamera2 not available and CLI camera init did not succeed"
            logger.warning(self.last_error)
            self._init_usb_camera()
            return

        requested_resolution = os.getenv("EXUVIA_PI_CAMERA_RES", "1920x1080")
        preferred_resolutions = [(1920, 1080), (1640, 1232), (1280, 720)]

        # Allow overriding the first attempt from environment, e.g. EXUVIA_PI_CAMERA_RES=4056x3040
        if "x" in requested_resolution.lower():
            try:
                w, h = requested_resolution.lower().split("x", 1)
                preferred_resolutions.insert(0, (int(w), int(h)))
            except ValueError:
                logger.warning(
                    "Invalid EXUVIA_PI_CAMERA_RES='%s'. Using safe defaults.",
                    requested_resolution,
                )

        for resolution in preferred_resolutions:
            try:
                self.picam2 = Picamera2(0)
                config = self.picam2.create_preview_configuration(
                    main={"size": resolution, "format": "RGB888"},
                    buffer_count=2,
                )
                self.picam2.configure(config)
                self.picam2.start()
                # Small warm-up helps avoid first-frame failures after reboot.
                time.sleep(0.2)
                self.camera_type = "pi"
                self.camera_resolution = resolution
                self.last_error = None
                logger.info("Pi HQ Camera initialized at %sx%s", resolution[0], resolution[1])
                return
            except Exception as e:
                self.last_error = f"Pi camera init failed at {resolution[0]}x{resolution[1]}: {e}"
                logger.warning(self.last_error)
                if self.picam2:
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                    self.picam2 = None

        logger.error("Failed to init Picamera2 at all resolutions. Falling back to USB.")
        self._init_usb_camera()

    def _init_rpicam_cli(self):
        """Initialize preferred camera path using Raspberry Pi CLI capture."""
        for cmd in ("rpicam-jpeg", "rpicam-still", "libcamera-still"):
            if shutil.which(cmd):
                self.rpicam_command = cmd
                break

        for cmd in ("rpicam-still", "libcamera-still"):
            if shutil.which(cmd):
                self.rpicam_preview_command = cmd
                break

        if self.rpicam_command is None:
            self.last_error = "Neither rpicam-still nor libcamera-still is available"
            logger.warning(self.last_error)
            return

        self.camera_type = "rpicam_cli"
        self.camera_resolution = (1280, 720)
        self.last_error = None
        logger.info("CLI camera fallback initialized using %s", self.rpicam_command)

    def _capture_rpicam_frame(self, width, height):
        """Capture one frame through rpicam/libcamera CLI and return RGB image."""
        if self.rpicam_command is None:
            return None

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        cmd = self._build_rpicam_command(output_path=output_path, width=width, height=height)

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "camera CLI command failed")

            frame_bgr = cv2.imread(output_path)
            if frame_bgr is None:
                raise RuntimeError("captured image could not be read")

            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            try:
                os.remove(output_path)
            except Exception:
                pass

    def _build_rpicam_command(self, output_path, width, height):
        """Build the CLI camera command for this Pi image stack."""
        return [
            self.rpicam_command,
            "-n",
            "--immediate",
            "--width",
            str(width),
            "--height",
            str(height),
            "-o",
            output_path,
        ]

    def _build_rpicam_preview_command(self, width, height, fps):
        """Build a persistent preview command that updates one JPEG repeatedly."""
        interval_ms = max(int(1000 / max(fps, 1)), 50)
        return [
            self.rpicam_preview_command,
            "-n",
            "-t",
            "0",
            "--timelapse",
            str(interval_ms),
            "--width",
            str(width),
            "--height",
            str(height),
            # Force full sensor field of view — without this, libcamera may
            # select a cropped sensor mode for lower resolutions.
            "--roi",
            "0,0,1,1",
            "-o",
            str(self.preview_path),
        ]

    def _get_capture_resolution(self):
        """Get full-resolution capture size for saved images."""
        requested_resolution = os.getenv("EXUVIA_PI_CAPTURE_RES", "4056x3040")
        try:
            width_str, height_str = requested_resolution.lower().split("x", 1)
            return int(width_str), int(height_str)
        except ValueError:
            logger.warning(
                "Invalid EXUVIA_PI_CAPTURE_RES='%s'. Falling back to 4056x3040.",
                requested_resolution,
            )
            return 4056, 3040

    def _get_preview_resolution(self):
        """Get preview size that preserves the full 4:3 field of view."""
        requested_resolution = os.getenv("EXUVIA_PI_PREVIEW_RES", "1920x1440")
        try:
            width_str, height_str = requested_resolution.lower().split("x", 1)
            return int(width_str), int(height_str)
        except ValueError:
            logger.warning(
                "Invalid EXUVIA_PI_PREVIEW_RES='%s'. Falling back to 1920x1440.",
                requested_resolution,
            )
            return 1920, 1440

    def start_preview_stream(self, fps=15, resolution=None):
        """Start persistent preview process for Pi camera live feed."""
        if self.camera_type != "rpicam_cli" or self.rpicam_preview_command is None:
            return False

        if resolution is None:
            preview_width, preview_height = self._get_preview_resolution()
        else:
            preview_width, preview_height = resolution

        requested_config = {
            "fps": fps,
            "resolution": (preview_width, preview_height),
        }

        if self.preview_process and self.preview_process.poll() is None:
            if self.preview_config == requested_config:
                return True
            self.stop_preview_stream()

        self.preview_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_rpicam_preview_command(preview_width, preview_height, fps)

        try:
            self.preview_log_handle = open(self.preview_dir / "preview.log", "ab")
            self.preview_process = subprocess.Popen(
                cmd,
                stdout=self.preview_log_handle,
                stderr=self.preview_log_handle,
            )
            time.sleep(0.6)

            if self.preview_process.poll() is not None:
                self.last_error = "Preview process exited immediately"
                logger.error(self.last_error)
                self.stop_preview_stream()
                return False

            self.preview_config = requested_config
            self.last_error = None
            logger.info(
                "Preview stream started with %s at %sx%s %sfps",
                self.rpicam_preview_command,
                preview_width,
                preview_height,
                fps,
            )
            return True
        except Exception as e:
            self.last_error = f"Failed to start preview stream: {e}"
            logger.error(self.last_error)
            self.stop_preview_stream()
            return False

    def stop_preview_stream(self):
        """Stop persistent preview process if running."""
        if self.preview_process:
            try:
                if self.preview_process.poll() is None:
                    self.preview_process.terminate()
                    self.preview_process.wait(timeout=2)
            except Exception:
                try:
                    self.preview_process.kill()
                except Exception:
                    pass
            finally:
                self.preview_process = None

        self.preview_config = None

        if self.preview_log_handle:
            try:
                self.preview_log_handle.close()
            except Exception:
                pass
            finally:
                self.preview_log_handle = None

    def is_preview_running(self):
        """Return whether the persistent preview process is active."""
        return self.preview_process is not None and self.preview_process.poll() is None

    def get_preview_jpeg(self):
        """Return the latest preview frame as raw JPEG bytes, with fallback to last valid frame.
        
        Validates JPEG integrity (FF D8 … FF D9 markers) before returning.
        Falls back to a cached copy of the last valid frame so the UI never goes blank
        due to a mid-write race condition.
        """
        if self.camera_type != "rpicam_cli":
            return None

        for _ in range(3):
            try:
                data = self.preview_path.read_bytes()
                # Only accept intact JPEG: must start with FF D8 and end with FF D9
                if len(data) > 100 and data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9':
                    self._last_preview_jpeg = data
                    return data
            except (IOError, OSError):
                pass
            time.sleep(0.02)

        # Return last known-good frame so the display doesn't flash blank
        return self._last_preview_jpeg

    def get_preview_frame(self, zoom=1):
        """Read the latest frame generated by the persistent preview process."""
        if self.camera_type != "rpicam_cli":
            return self.get_frame(zoom=zoom)

        if not self.is_preview_running() and not self.start_preview_stream(fps=10):
            return self.last_frame  # return last good numpy frame if available

        for _ in range(5):
            frame_bgr = cv2.imread(str(self.preview_path))
            if frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if zoom > 1:
                    frame_rgb = self._apply_zoom(frame_rgb, zoom)
                self.last_frame = frame_rgb
                return frame_rgb
            time.sleep(0.05)

        logger.warning("Preview frame unreadable, returning last good frame")
        return self.last_frame
    
    def _init_usb_camera(self):
        """Initialize USB webcam fallback"""
        attempts = [(0, cv2.CAP_V4L2), (0, cv2.CAP_ANY), (1, cv2.CAP_V4L2), (1, cv2.CAP_ANY)]

        for index, backend in attempts:
            try:
                self.cap = cv2.VideoCapture(index, backend)
                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open webcam index={index}, backend={backend}")

                # Lower defaults for Pi stability; can still upscale in UI.
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.camera_type = "usb"
                self.camera_resolution = (actual_w, actual_h)
                self.last_error = None
                logger.info(
                    "USB webcam initialized index=%s backend=%s at %sx%s",
                    index,
                    backend,
                    actual_w,
                    actual_h,
                )
                return
            except Exception as e:
                self.last_error = f"USB camera init failed index={index}, backend={backend}: {e}"
                logger.warning(self.last_error)
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None

        logger.error("Failed to init USB camera")
        self.camera_type = None
    
    def get_frame(self, zoom=1):
        """
        Get current frame with optional zoom.
        
        Args:
            zoom: 1, 2, or 3 for zoom level
        
        Returns:
            numpy array (RGB format) or None
        """
        frame = None
        
        try:
            if self.camera_type == "pi" and self.picam2:
                frame = self.picam2.capture_array()
                if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Already RGB
                    pass
            
            elif self.camera_type == "usb" and self.cap:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif self.camera_type == "rpicam_cli":
                frame = self._capture_rpicam_frame(1280, 720)
            
            if frame is not None and zoom > 1:
                frame = self._apply_zoom(frame, zoom)
                self.current_zoom = zoom
            
            self.last_frame = frame
            return frame
        
        except Exception as e:
            self.last_error = f"Error getting frame: {e}"
            logger.error(self.last_error)
            return None
    
    def _apply_zoom(self, frame, zoom):
        """Crop and upscale for digital zoom"""
        h, w = frame.shape[:2]
        crop_h, crop_w = h // zoom, w // zoom
        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2
        cropped = frame[y1:y1+crop_h, x1:x1+crop_w]
        return cv2.resize(cropped, (w, h))
    
    def capture_image(self, tray_id, save_dir=None, zoom=1):
        """
        Capture and save image.
        
        Args:
            tray_id: Label for the tray (e.g., "tray_001")
            save_dir: Directory to save (default: data/captures)
            zoom: Current zoom level to record
        
        Returns:
            Path to saved image or None
        """
        if save_dir is None:
            save_dir = Path("data/captures")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{tray_id}_{timestamp}_z{zoom}x.jpg"
            filepath = save_dir / filename

            if self.camera_type == "rpicam_cli" and self.rpicam_command:
                was_preview_running = self.is_preview_running()
                preview_config = self.preview_config.copy() if self.preview_config else None
                if was_preview_running:
                    self.stop_preview_stream()

                capture_width, capture_height = self._get_capture_resolution()
                cmd = self._build_rpicam_command(
                    output_path=str(filepath),
                    width=capture_width,
                    height=capture_height,
                )
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=12,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or "camera capture command failed")

                if zoom > 1:
                    frame_bgr = cv2.imread(str(filepath))
                    if frame_bgr is None:
                        raise RuntimeError("captured image could not be read")
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    zoomed_rgb = self._apply_zoom(frame_rgb, zoom)
                    zoomed_bgr = cv2.cvtColor(zoomed_rgb, cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(filepath), zoomed_bgr):
                        raise RuntimeError("zoomed image could not be written")

                if was_preview_running:
                    self.start_preview_stream(
                        fps=preview_config["fps"] if preview_config else 15,
                        resolution=preview_config["resolution"] if preview_config else None,
                    )

                logger.info("Image saved via %s: %s", self.rpicam_command, filepath)
                return filepath

            frame = self.get_frame(zoom)
            if frame is None:
                logger.error("No frame to capture")
                return None
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(filepath), frame_bgr)
            
            if success:
                logger.info(f"Image saved: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to write image: {filepath}")
                return None
        
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None
    
    def capture_tray_jpeg(self):
        """Capture one full-resolution still to a temp file using the active Pi camera CLI.

        Stops the live preview stream while capturing, restarts it afterwards.
        Returns the Path to the JPEG temp file, or None on failure.
        The caller is responsible for deleting the file when done with it.
        """
        if self.camera_type != "rpicam_cli" or self.rpicam_command is None:
            self.last_error = "rpicam_cli camera not available for tray capture"
            logger.error(self.last_error)
            return None

        self.preview_dir.mkdir(parents=True, exist_ok=True)
        import tempfile as _tf
        tmp = _tf.NamedTemporaryFile(
            suffix=".jpg", prefix="tray_preview_", delete=False,
            dir=str(self.preview_dir),
        )
        tmp.close()
        output_path = Path(tmp.name)

        was_preview_running = self.is_preview_running()
        saved_config = self.preview_config.copy() if self.preview_config else None
        if was_preview_running:
            self.stop_preview_stream()

        capture_width, capture_height = self._get_capture_resolution()
        cmd = self._build_rpicam_command(
            output_path=str(output_path),
            width=capture_width,
            height=capture_height,
        )
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or f"{self.rpicam_command} failed")
            if not output_path.exists():
                raise RuntimeError(f"{self.rpicam_command} produced no output file")
            logger.info(
                "Tray JPEG captured via %s %sx%s -> %s",
                self.rpicam_command,
                capture_width,
                capture_height,
                output_path,
            )
            return output_path
        except Exception as exc:
            self.last_error = f"Tray capture failed: {exc}"
            logger.error(self.last_error)
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        finally:
            if was_preview_running and saved_config:
                self.start_preview_stream(
                    fps=saved_config["fps"],
                    resolution=saved_config["resolution"],
                )

    def close(self):
        """Clean up resources"""
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
        self.stop_preview_stream()
        if self.cap:
            self.cap.release()
        logger.info("Camera closed")
    
    def is_available(self):
        """Check if camera is available"""
        return self.camera_type is not None

    def get_status(self):
        """Return camera status for UI diagnostics"""
        return {
            "type": self.camera_type,
            "resolution": self.camera_resolution,
            "last_error": self.last_error,
            "preview_running": self.is_preview_running(),
            "preview_config": self.preview_config,
        }


# Singleton for Streamlit (prevents reinitialization)
_camera_instance = None

def get_camera(use_pi_camera=True, force_reinit=False):
    """Get or create camera manager"""
    global _camera_instance
    if force_reinit and _camera_instance is not None:
        try:
            _camera_instance.close()
        except Exception:
            pass
        _camera_instance = None

    if _camera_instance is None:
        _camera_instance = CameraManager(use_pi_camera=use_pi_camera)
    return _camera_instance


def reset_camera(use_pi_camera=True):
    """Force close and recreate camera singleton."""
    return get_camera(use_pi_camera=use_pi_camera, force_reinit=True)
