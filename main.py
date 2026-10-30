from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def quad_ok(proj: np.ndarray, shape: tuple[int, ...]) -> bool:
    pts = proj.reshape(4, 2).astype(np.float32)
    if cv2.contourArea(pts, oriented=True) <= 1:
        return False
    contour = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False
    height, width = shape[:2]
    area = cv2.contourArea(pts)
    if area < 0.0015 * width * height or area > 2.5 * width * height:
        return False
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    if min(sides) < 8 or max(sides) > 5 * min(sides):
        return False
    frame = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    visible, _ = cv2.intersectConvexConvex(pts, frame)
    return visible >= 0.6 * area


def span_ok(pts: np.ndarray, shape: tuple[int, ...]) -> bool:
    height, width = shape[:2]
    span = pts.max(axis=0) - pts.min(axis=0)
    return bool(span[0] > 0.28 * width and span[1] > 0.18 * height)


def extent_ok(quad: np.ndarray, ref_pts: np.ndarray, dst_pts: np.ndarray, ref_shape: tuple[int, ...]) -> bool:
    height, width = ref_shape[:2]
    ref_span = np.maximum(ref_pts.max(axis=0) - ref_pts.min(axis=0), 1.0)
    dst_span = np.maximum(dst_pts.max(axis=0) - dst_pts.min(axis=0), 1.0)
    quad_span = quad.max(axis=0) - quad.min(axis=0)
    expect = dst_span * np.array([width, height], np.float32) / ref_span
    ratio = quad_span / expect
    return bool(np.all(ratio > 0.7) and np.all(ratio < 1.4))


def fit_quad(src, dst, corners, frame_shape, ref_shape, min_inliers: int):
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None or mask is None:
        return None
    keep = mask.ravel().astype(bool)
    if int(keep.sum()) < min_inliers:
        return None
    ref_pts = src.reshape(-1, 2)[keep]
    if not span_ok(ref_pts, ref_shape):
        return None
    projected = cv2.perspectiveTransform(corners, homography)
    quad = projected.reshape(4, 2)
    dst_pts = dst.reshape(-1, 2)[keep]
    if not quad_ok(projected, frame_shape) or not extent_ok(quad, ref_pts, dst_pts, ref_shape):
        return None
    return quad, ref_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), homography


def orb_fit(orb, matcher, kp_ref, des_ref, gray, corners, ref_shape, min_inliers: int):
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 4:
        return None
    pairs = matcher.knnMatch(des_ref, descriptors, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)
    if len(good) < min_inliers:
        return None
    src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    fitted = fit_quad(src, dst, corners, gray.shape, ref_shape, min_inliers)
    affine = affine_quad(src, dst, corners, gray.shape)
    if fitted is not None and (affine is None or keystone(fitted[0]) <= 1.8):
        return fitted
    return affine


def keystone(quad: np.ndarray) -> float:
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    opposite = (
        max(sides[0], sides[2]) / max(min(sides[0], sides[2]), 1.0),
        max(sides[1], sides[3]) / max(min(sides[1], sides[3]), 1.0),
    )
    return float(max(opposite))


def affine_quad(src, dst, corners, frame_shape):
    if len(src) < 6:
        return None
    matrix, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if matrix is None or mask is None or int(mask.sum()) < 8:
        return None
    projected = cv2.transform(corners, matrix)
    if not quad_ok(projected, frame_shape):
        return None
    keep = mask.ravel().astype(bool)
    return projected.reshape(4, 2), src[keep], dst[keep], matrix


def align(reference_gray, gray, warp, corners):
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    try:
        score, updated = cv2.findTransformECC(
            reference_gray, gray, warp.copy(), cv2.MOTION_AFFINE, criteria, None, 5
        )
    except cv2.error:
        return None
    quad = cv2.transform(corners, updated).reshape(4, 2)
    if not quad_ok(quad, gray.shape):
        return None
    display = quad
    area = cv2.contourArea(quad.astype(np.float32))
    if area > 0.05 * gray.shape[0] * gray.shape[1]:
        homography = np.vstack([updated, [0, 0, 1]]).astype(np.float32)
        try:
            refined, matrix = cv2.findTransformECC(
                reference_gray,
                gray,
                homography,
                cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-4),
                None,
                5,
            )
        except cv2.error:
            refined = None
        if refined is not None and refined >= score + 0.01:
            tilted = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)
            tilted_area = cv2.contourArea(tilted.astype(np.float32))
            ratio = tilted_area / max(area, 1.0)
            shift = float(np.linalg.norm(tilted - quad, axis=1).mean())
            side = float(np.mean([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]))
            if (
                quad_ok(tilted, gray.shape)
                and 1.12 <= keystone(tilted) <= 2.0
                and 0.8 <= ratio <= 1.25
                and shift <= 0.25 * side
            ):
                display = tilted
    return float(score), updated, display


def steady(quad: np.ndarray, previous: np.ndarray | None) -> bool:
    if previous is None:
        return True
    shift = float(np.linalg.norm(quad - previous, axis=1).mean())
    area = cv2.contourArea(quad.astype(np.float32))
    previous_area = cv2.contourArea(previous.astype(np.float32))
    ratio = area / max(previous_area, 1.0)
    return shift < 45 and 0.8 <= ratio <= 1.25


def calm(quad: np.ndarray, previous: np.ndarray | None, corners: np.ndarray) -> np.ndarray:
    if previous is None:
        return quad
    shift = float(np.linalg.norm(quad - previous, axis=1).mean())
    if shift > 50:
        weight = 1.0
    elif shift > 20:
        weight = 0.6
    else:
        weight = 0.35
    blended = weight * quad + (1.0 - weight) * previous
    if keystone(quad) <= 1.08:
        matrix, _ = cv2.estimateAffine2D(corners, blended.reshape(-1, 1, 2))
        if matrix is not None:
            return cv2.transform(corners, matrix).reshape(4, 2)
    if cv2.contourArea(blended.astype(np.float32), oriented=True) <= 1:
        return quad
    contour = np.round(blended).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return quad
    return blended


def draw_box(frame: np.ndarray, quad: np.ndarray, label: str) -> None:
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, (36, 190, 60), 2, cv2.LINE_AA)
    flat = pts.reshape(-1, 2)
    x = int(np.clip(flat[:, 0].min(), 8, frame.shape[1] - 90))
    y = int(np.clip(flat[:, 1].min() - 8, 22, frame.shape[0] - 8))
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 190, 60), 2, cv2.LINE_AA)


def track_video(src: Path, dst: Path, label: str = "object") -> dict:
    capture = cv2.VideoCapture(str(src))
    ok, reference = capture.read()
    if not ok:
        raise SystemExit(f"cannot read {src}")
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=10)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    if des_ref is None:
        raise SystemExit("no keypoints on the first frame")
    height, width = gray_ref.shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    writer.write(reference)
    quads: list[np.ndarray | None] = [None]
    detected = 0
    shown = None
    lost = 0
    hold = 8
    warp = np.eye(2, 3, dtype=np.float32)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chosen = None
        from_ecc = False
        aligned = align(gray_ref, gray, warp, corners)
        if aligned is not None and aligned[0] >= 0.55 and (aligned[0] >= 0.70 or steady(aligned[2], shown)):
            chosen = aligned[2]
            warp = aligned[1]
            from_ecc = True
        else:
            fresh = lost >= hold
            locked = orb_fit(orb, matcher, kp_ref, des_ref, gray, corners, reference.shape, 20 if fresh else 12)
            if locked is not None and (fresh or steady(locked[0], shown) or len(locked[1]) >= 40):
                chosen = locked[0]
        if chosen is None:
            lost += 1
            if shown is not None and lost <= hold:
                detected += 1
                draw_box(frame, shown, label)
                quads.append(shown.copy())
            else:
                quads.append(None)
        else:
            quad = calm(chosen, shown if lost == 0 else None, corners)
            if not from_ecc:
                matrix, _ = cv2.estimateAffine2D(corners, quad.reshape(-1, 1, 2))
                if matrix is not None:
                    warp = matrix
            shown = quad
            lost = 0
            detected += 1
            draw_box(frame, quad, label)
            quads.append(quad.copy())
        writer.write(frame)
    capture.release()
    writer.release()
    return {"frames": len(quads) - 1, "detected": detected, "quads": quads}


def main() -> None:
    parser = argparse.ArgumentParser(description="Track a planar object. The first frame is the object.")
    parser.add_argument("video")
    parser.add_argument("-o", "--output")
    parser.add_argument("--label", default="object")
    args = parser.parse_args()
    src = Path(args.video)
    dst = Path(args.output) if args.output else Path("output") / f"{src.stem}_tracked.mp4"
    stats = track_video(src, dst, args.label)
    print(f"scene frames {stats['frames']}  with box {stats['detected']}  saved {dst}")


if __name__ == "__main__":
    main()
