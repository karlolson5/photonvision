/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipe.impl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.wpilib.vision.apriltag.AprilTagDetection;

/**
 * Tracks recently-seen AprilTag locations across frames so {@link
 * org.photonvision.vision.pipeline.AprilTagPipeline} can skip the NPU-based ROI detection stage
 * ({@link AprilTagROIDetectionPipe}) on frames where tags are already being tracked, and instead
 * decode only the small regions where those tags are expected to still be.
 *
 * <p>This class never touches the NPU itself. It just remembers, per tag ID, the bounding box
 * from the last frame that tag was confirmed on, plus a smoothed frame-to-frame velocity estimate
 * of that box's center. {@link #predict} hands those boxes back out (nudged by velocity, then
 * padded) as candidate search regions for {@link AprilTagROIDecodePipe}, which is cheap CPU-only
 * work and doesn't care whether its input ROIs came from a neural net or from here.
 *
 * <p>Equivalent in spirit to the "temporal acceleration" search-region prediction used by the
 * EagleEye Vision System, adapted to key off decoded tag IDs (since PhotonVision decodes tags in
 * the same pass it would otherwise use to detect ROIs) rather than a separate motion model.
 *
 * <p>Not thread-safe. Intended for use by a single {@code AprilTagPipeline} instance, i.e. one
 * instance per camera.
 */
public class TagRoiTracker {
    /** Per-tag tracking state. */
    private static class TrackedTag {
        RotatedRect lastRoi;
        double velocityX; // px/frame, exponentially smoothed
        double velocityY;
        int framesSinceConfirmed = 0;

        TrackedTag(RotatedRect roi) {
            this.lastRoi = roi;
        }
    }

    // How much a new measurement moves the velocity estimate vs. trusting the existing one.
    // Kept simple and fixed rather than exposed as a setting -- this is not sensitive enough to
    // need per-pipeline tuning, and the retry-on-miss behavior below covers most of the risk of
    // getting it slightly wrong.
    private static final double VELOCITY_SMOOTHING = 0.5;

    private final Map<Integer, TrackedTag> trackedTags = new HashMap<>();
    private int framesSinceFullScan = 0;
    private boolean forceRescanNextFrame = true;

    /**
     * @param maxCoastFrames how many consecutive frames a tag may go unconfirmed before it's
     *     dropped and a full NPU rescan is required to reacquire it
     * @param rescanIntervalFrames force a full NPU rescan at least this often even when tracking
     *     looks healthy, as a safety net against slowly-compounding prediction error
     * @return true if {@link #predict} should be used this frame instead of running the full
     *     ML-assisted (NPU) detection pass
     */
    public boolean shouldUseLookahead(int maxCoastFrames, int rescanIntervalFrames) {
        if (forceRescanNextFrame || trackedTags.isEmpty()) {
            return false;
        }
        if (framesSinceFullScan >= rescanIntervalFrames) {
            return false;
        }
        for (TrackedTag t : trackedTags.values()) {
            if (t.framesSinceConfirmed > maxCoastFrames) {
                return false;
            }
        }
        return true;
    }

    /**
     * Builds predicted, padded search ROIs for every currently tracked tag. Intended to be passed
     * straight to {@link AprilTagROIDecodePipe} in place of the NPU detection stage's output.
     */
    public List<RotatedRect> predict(int frameWidth, int frameHeight, int paddingPixels) {
        List<RotatedRect> rois = new ArrayList<>(trackedTags.size());
        for (TrackedTag t : trackedTags.values()) {
            RotatedRect predicted =
                    new RotatedRect(
                            new Point(t.lastRoi.center.x + t.velocityX, t.lastRoi.center.y + t.velocityY),
                            t.lastRoi.size,
                            t.lastRoi.angle);
            rois.add(
                    AprilTagROIDecodePipe.expandBbox(predicted, paddingPixels, frameWidth, frameHeight));
        }
        return rois;
    }

    /**
     * Call after a full ML-assisted (NPU) detection + decode pass. Replaces tracking state
     * wholesale from the decoded tags, since that's a ground-truth reacquisition.
     */
    public void updateFromFullScan(List<AprilTagDetection> detections) {
        trackedTags.clear();
        for (AprilTagDetection det : detections) {
            trackedTags.put(det.getId(), new TrackedTag(boundingBoxOf(det)));
        }
        framesSinceFullScan = 0;
        forceRescanNextFrame = false;
    }

    /**
     * Call after a lookahead (decode-only, no NPU) pass. Updates position/velocity for confirmed
     * tags, ages out tags that weren't found in their predicted ROI, and requests a full rescan on
     * the next frame once a tag has gone unconfirmed for more than {@code maxCoastFrames}
     * consecutive frames.
     *
     * @param maxCoastFrames how many consecutive missed frames a tag may tolerate before it's
     *     treated as lost and a full rescan is requested. Pass the same value used in {@link
     *     #shouldUseLookahead} so the two stay consistent -- a single miss no longer forces an
     *     immediate rescan by itself, since a brief partial occlusion is exactly the case this
     *     budget exists to tolerate.
     */
    public void updateFromLookahead(List<AprilTagDetection> detections, int maxCoastFrames) {
        Map<Integer, AprilTagDetection> byId = new HashMap<>();
        for (AprilTagDetection det : detections) {
            byId.put(det.getId(), det);
        }

        boolean anyExceededCoastBudget = false;
        for (Map.Entry<Integer, TrackedTag> entry : trackedTags.entrySet()) {
            TrackedTag tracked = entry.getValue();
            AprilTagDetection det = byId.get(entry.getKey());
            if (det == null) {
                tracked.framesSinceConfirmed++;
                if (tracked.framesSinceConfirmed > maxCoastFrames) {
                    anyExceededCoastBudget = true;
                }
                continue;
            }

            RotatedRect newRoi = boundingBoxOf(det);
            double newVx = newRoi.center.x - tracked.lastRoi.center.x;
            double newVy = newRoi.center.y - tracked.lastRoi.center.y;
            tracked.velocityX = VELOCITY_SMOOTHING * newVx + (1 - VELOCITY_SMOOTHING) * tracked.velocityX;
            tracked.velocityY = VELOCITY_SMOOTHING * newVy + (1 - VELOCITY_SMOOTHING) * tracked.velocityY;
            tracked.lastRoi = newRoi;
            tracked.framesSinceConfirmed = 0;
        }

        framesSinceFullScan++;
        if (anyExceededCoastBudget) {
            forceRescanNextFrame = true;
        }
    }

    /** Drops all tracking state, e.g. when ML detection is disabled, unavailable, or reset. */
    public void reset() {
        trackedTags.clear();
        framesSinceFullScan = 0;
        forceRescanNextFrame = true;
    }

    private static RotatedRect boundingBoxOf(AprilTagDetection det) {
        double minX = Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            double x = det.getCornerX(i);
            double y = det.getCornerY(i);
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
        return new RotatedRect(
                new Point((minX + maxX) / 2.0, (minY + maxY) / 2.0),
                new Size(maxX - minX, maxY - minY),
                0);
    }
}
