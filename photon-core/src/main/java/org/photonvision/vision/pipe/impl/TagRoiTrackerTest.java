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

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.opencv.core.RotatedRect;
import org.wpilib.vision.apriltag.AprilTagDetection;

/**
 * Tests for {@link TagRoiTracker} in isolation. This class has no JNI/NPU dependency -- it only
 * ever consumes already-decoded {@link AprilTagDetection}s -- so these run on any host, including
 * CI machines without a Rubik Pi 3 or MemryX card attached.
 */
public class TagRoiTrackerTest {
    private static final int FRAME_WIDTH = 640;
    private static final int FRAME_HEIGHT = 480;

    // Identity-ish placeholder; boundingBoxOf() (and therefore everything under test here) never
    // reads the homography, so its contents don't matter for these tests.
    private static final double[] DUMMY_HOMOGRAPHY = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    /** Builds a square, axis-aligned AprilTagDetection centered at (centerX, centerY). */
    private static AprilTagDetection makeDetection(
            int id, double centerX, double centerY, double halfSize) {
        double[] corners = {
            centerX - halfSize, centerY - halfSize, // top-left
            centerX + halfSize, centerY - halfSize, // top-right
            centerX + halfSize, centerY + halfSize, // bottom-right
            centerX - halfSize, centerY + halfSize, // bottom-left
        };
        return new AprilTagDetection(
                "tag36h11", id, 0, 100.0, DUMMY_HOMOGRAPHY, centerX, centerY, corners);
    }

    @Test
    public void freshTrackerRequiresFullScan() {
        var tracker = new TagRoiTracker();

        // No full scan has ever run -- forceRescanNextFrame starts true specifically so the very
        // first frame always goes through the NPU rather than trying to predict from nothing.
        assertFalse(tracker.shouldUseLookahead(6, 30));
    }

    @Test
    public void fullScanWithNoDetectionsLeavesLookaheadUnavailable() {
        var tracker = new TagRoiTracker();

        tracker.updateFromFullScan(List.of());

        // forceRescanNextFrame is cleared, but there's nothing to predict -- the emptiness check
        // in shouldUseLookahead should still block lookahead.
        assertFalse(tracker.shouldUseLookahead(6, 30));
    }

    @Test
    public void fullScanWithDetectionsEnablesLookahead() {
        var tracker = new TagRoiTracker();

        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        assertTrue(tracker.shouldUseLookahead(6, 30));
    }

    @Test
    public void predictReturnsPaddedBoxAroundLastKnownPosition() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        List<RotatedRect> rois = tracker.predict(FRAME_WIDTH, FRAME_HEIGHT, 20);

        assertEquals(1, rois.size());
        RotatedRect roi = rois.get(0);
        // Tag's bounding box is 20x20 (halfSize 10); +20px padding on each side -> 60x60.
        assertEquals(60.0, roi.size.width, 1e-6);
        assertEquals(60.0, roi.size.height, 1e-6);
        // No velocity yet (this is the first prediction after a full scan), so the predicted
        // center should sit exactly on the tag's last known position.
        assertEquals(100.0, roi.center.x, 1e-6);
        assertEquals(100.0, roi.center.y, 1e-6);
    }

    @Test
    public void predictClampsPaddedBoxToFrameBounds() {
        var tracker = new TagRoiTracker();
        // Tag right in the corner of the frame.
        tracker.updateFromFullScan(List.of(makeDetection(1, 5, 5, 10)));

        List<RotatedRect> rois = tracker.predict(FRAME_WIDTH, FRAME_HEIGHT, 20);

        RotatedRect roi = rois.get(0);
        assertTrue(roi.center.x - roi.size.width / 2.0 >= -1e-6, "ROI should not extend past x=0");
        assertTrue(roi.center.y - roi.size.height / 2.0 >= -1e-6, "ROI should not extend past y=0");
    }

    @Test
    public void confirmedLookaheadFrameResetsCoastCounterAndKeepsTracking() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        // Tag confirmed again in roughly the same spot.
        tracker.updateFromLookahead(List.of(makeDetection(1, 101, 100, 10)), 6);

        assertTrue(tracker.shouldUseLookahead(6, 30));
    }

    @Test
    public void movingTagShiftsPredictionByEstimatedVelocity() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        // Tag moved +20px in x between the full scan and this lookahead frame.
        tracker.updateFromLookahead(List.of(makeDetection(1, 120, 100, 10)), 6);

        List<RotatedRect> rois = tracker.predict(FRAME_WIDTH, FRAME_HEIGHT, 0);
        // Velocity smoothing is 0.5, applied on top of a starting velocity of 0, so the estimated
        // velocity is 0.5 * 20 = 10px/frame; predict() applies that on top of the tag's last
        // confirmed position (120), landing at 130.
        assertEquals(130.0, rois.get(0).center.x, 1e-6);
    }

    @Test
    public void singleMissWithinCoastBudgetDoesNotForceRescan() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        // Tag not found in its predicted ROI this frame, but the coast budget is 3 frames.
        tracker.updateFromLookahead(List.of(), 3);

        assertTrue(
                tracker.shouldUseLookahead(3, 30),
                "A single miss within the coast budget should not force a full rescan");
    }

    @Test
    public void missExceedingCoastBudgetForcesRescan() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        int maxCoastFrames = 3;
        // Miss for maxCoastFrames + 1 consecutive frames to push framesSinceConfirmed past budget.
        for (int i = 0; i < maxCoastFrames + 1; i++) {
            tracker.updateFromLookahead(List.of(), maxCoastFrames);
        }

        assertFalse(
                tracker.shouldUseLookahead(maxCoastFrames, 30),
                "Exceeding the coast budget should force a full rescan");
    }

    @Test
    public void rescanIntervalForcesPeriodicFullScanEvenWithoutMisses() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        int rescanIntervalFrames = 5;
        // Confirm the same tag every frame -- tracking never lapses -- but run enough lookahead
        // frames to hit the periodic rescan safety net.
        for (int i = 0; i < rescanIntervalFrames; i++) {
            tracker.updateFromLookahead(List.of(makeDetection(1, 100, 100, 10)), 6);
        }

        assertFalse(
                tracker.shouldUseLookahead(6, rescanIntervalFrames),
                "A periodic rescan should be forced even when every frame confirmed cleanly");
    }

    @Test
    public void resetClearsTrackingState() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));
        assertTrue(tracker.shouldUseLookahead(6, 30));

        tracker.reset();

        assertFalse(tracker.shouldUseLookahead(6, 30));
        assertTrue(tracker.predict(FRAME_WIDTH, FRAME_HEIGHT, 20).isEmpty());
    }

    @Test
    public void newFullScanReplacesStaleTrackedTags() {
        var tracker = new TagRoiTracker();
        tracker.updateFromFullScan(List.of(makeDetection(1, 100, 100, 10)));

        // A second full scan sees a different tag entirely (tag 1 is gone, tag 2 appeared).
        tracker.updateFromFullScan(List.of(makeDetection(2, 300, 300, 10)));

        List<RotatedRect> rois = tracker.predict(FRAME_WIDTH, FRAME_HEIGHT, 0);
        assertEquals(1, rois.size(), "Old tag should not still be tracked after a fresh full scan");
        assertEquals(300.0, rois.get(0).center.x, 1e-6);
    }
}
