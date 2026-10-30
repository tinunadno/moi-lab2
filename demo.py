from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from main import draw_box, track_video

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
WIDTH = 640
HEIGHT = 480
COUNT = 61


def texture(rng: np.random.Generator) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 245, np.uint8)
    for _ in range(50):
        x = int(rng.integers(0, WIDTH - 40))
        y = int(rng.integers(0, HEIGHT - 30))
        w = int(rng.integers(16, 70))
        h = int(rng.integers(10, 40))
        color = tuple(int(v) for v in rng.integers(0, 180, 3))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for row in range(10):
        for col in range(16):
            char = alphabet[int(rng.integers(0, len(alphabet)))]
            shade = int(rng.integers(0, 40))
            origin = (12 + col * 39, 36 + row * 44)
            cv2.putText(image, char, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (shade, shade, shade), 2, cv2.LINE_AA)
    return image


def background() -> np.ndarray:
    column = np.linspace(198, 228, HEIGHT, dtype=np.float32)[:, None]
    row = np.linspace(0, 14, WIDTH, dtype=np.float32)[None, :]
    plane = np.clip(column + row, 0, 255).astype(np.uint8)
    return cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)


def placed(cx: float, cy: float, scale: float, angle: float, perspective: float) -> np.ndarray:
    ow = WIDTH * scale
    oh = HEIGHT * scale
    local = np.float32([[-ow / 2, -oh / 2], [ow / 2, -oh / 2], [ow / 2, oh / 2], [-ow / 2, oh / 2]])
    shrink = 1 - perspective
    local[0, 0] *= shrink
    local[1, 0] *= shrink
    local[0, 1] -= oh * perspective * 0.2
    local[1, 1] -= oh * perspective * 0.2
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.float32([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.float32([cx, cy])


def paste(scene: np.ndarray, obj: np.ndarray, quad: np.ndarray) -> None:
    src = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    homography = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(obj, homography, (WIDTH, HEIGHT))
    mask = cv2.warpPerspective(np.full((HEIGHT, WIDTH), 255, np.uint8), homography, (WIDTH, HEIGHT))
    scene[mask > 128] = warped[mask > 128]


def shade(frame: np.ndarray, t: float) -> np.ndarray:
    alpha = 1.0 + 0.12 * np.sin(2 * np.pi * t)
    beta = 16 * np.sin(2 * np.pi * t)
    return cv2.convertScaleAbs(frame, alpha=float(alpha), beta=float(beta))


def cover(frame: np.ndarray, quad: np.ndarray, frac: float) -> None:
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    cut = x0 + (x1 - x0) * frac
    cv2.rectangle(frame, (int(x0) - 4, int(y0) - 4), (int(cut), int(y1) + 4), (80, 96, 120), -1)


def visible_fraction(quad: np.ndarray) -> float:
    area = cv2.contourArea(quad.astype(np.float32))
    if area <= 1:
        return 0.0
    frame = np.float32([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]])
    inside, _ = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    return float(inside / area)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left = np.zeros((HEIGHT, WIDTH), np.uint8)
    right = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillConvexPoly(left, np.round(a).astype(np.int32), 255)
    cv2.fillConvexPoly(right, np.round(b).astype(np.int32), 255)
    inter = np.count_nonzero((left > 0) & (right > 0))
    union = np.count_nonzero((left > 0) | (right > 0))
    return inter / union if union else 0.0


def motion_pose(t: float) -> tuple[np.ndarray, float]:
    cx = WIDTH / 2 + WIDTH * 0.22 * np.sin(2 * np.pi * t)
    cy = HEIGHT / 2 + HEIGHT * 0.16 * np.cos(2 * np.pi * t)
    quad = placed(cx, cy, 0.38, -0.4 + 0.8 * t, 0.0)
    return quad, 0.0


def perspective_pose(t: float) -> tuple[np.ndarray, float]:
    scale = 0.30 + 0.24 * (0.5 + 0.5 * np.sin(2 * np.pi * t))
    tilt = 0.12 + 0.28 * (0.5 - 0.5 * np.cos(2 * np.pi * t))
    quad = placed(WIDTH / 2, HEIGHT / 2, scale, 0.18 * np.sin(2 * np.pi * t), tilt)
    return quad, 0.0


def occlusion_pose(t: float) -> tuple[np.ndarray, float]:
    cx = WIDTH * 0.32 + t * WIDTH * 0.95
    quad = placed(cx, HEIGHT / 2, 0.40, 0.12 * np.sin(2 * np.pi * t), 0.05)
    frac = 0.42 if 0.18 < t < 0.48 else 0.0
    return quad, frac


def build(name: str, obj: np.ndarray, pose) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    plate = background()
    frames = [obj.copy()]
    quads = []
    visible = []
    for index in range(COUNT):
        t = index / (COUNT - 1)
        quad, frac = pose(t)
        scene = shade(plate, t)
        paste(scene, obj, quad)
        if frac:
            cover(scene, quad, frac)
        frames.append(scene)
        quads.append(quad)
        visible.append(visible_fraction(quad) * (1 - frac))
    path = OUTPUT / f"{name}.mp4"
    write_video(path, frames)
    return frames, quads, visible


def write_video(path: Path, frames: list[np.ndarray], fps: float = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def titled(frame: np.ndarray, title: str) -> np.ndarray:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def evaluate(name: str, scenes: list[np.ndarray], truth: list[np.ndarray], visible: list[float], preview: int) -> np.ndarray:
    stats = track_video(OUTPUT / f"{name}.mp4", OUTPUT / f"{name}_tracked.mp4", "object")
    found = stats["quads"][1:]
    scores = []
    hits = 0
    false = 0
    should = 0
    for quad, seen, guess in zip(truth, visible, found):
        if seen >= 0.30:
            should += 1
            if guess is not None:
                score = iou(quad, guess)
                scores.append(score)
                if score >= 0.5:
                    hits += 1
        elif guess is not None and seen < 0.12:
            false += 1
    mean = float(np.mean(scores)) if scores else 0.0
    print(
        f"{name:12}  should {should:3d}  iou>=0.5 {hits:3d}  "
        f"mean IoU {mean:.3f}  false {false:3d}  boxes {stats['detected']:3d}"
    )
    shown = scenes[preview].copy()
    guess = found[preview - 1]
    if guess is not None:
        draw_box(shown, guess, "object")
    return titled(shown, name)


def main() -> None:
    obj = texture(np.random.default_rng(2))
    specs = (
        ("motion", motion_pose, 16),
        ("perspective", perspective_pose, 30),
        ("occlusion", occlusion_pose, 20),
    )
    panels = []
    for name, pose, preview in specs:
        scenes, truth, visible = build(name, obj, pose)
        panels.append(evaluate(name, scenes, truth, visible, preview))
    cv2.imwrite(str(OUTPUT / "results.png"), np.hstack(panels))
    print(f"saved {OUTPUT / 'results.png'}")


if __name__ == "__main__":
    main()
