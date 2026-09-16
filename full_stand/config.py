from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT: Path = Path(__file__).resolve().parents[1]
CHECKPOINTS: Path = ROOT / 'checkpoints'


@dataclass(slots=True)
class StandConfig:
    detector: Path = CHECKPOINTS / 'plate_detector.pt'
    recognizer: Path = CHECKPOINTS / 'plate_recognizer.pt'
    device: Optional[str] = None
    host: str = '127.0.0.1'
    port: int = 8080
    confidence: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    base_pad: float = 0.04
    max_detections: int = 64
    min_box_side: float = 6.0
    max_pixels: int = 120_000_000
    use_ema: bool = True
    use_graphs: bool = True
    use_script: bool = False
    top_alternatives: int = 4
    max_upload_bytes: int = 24 * 1024 * 1024
    history_limit: int = 400
    queue_limit: int = 512

    def validate(self) -> None:
        if not self.detector.is_file():
            raise FileNotFoundError(f'detector checkpoint not found: {self.detector}')
        if not self.recognizer.is_file():
            raise FileNotFoundError(f'recognizer checkpoint not found: {self.recognizer}')
