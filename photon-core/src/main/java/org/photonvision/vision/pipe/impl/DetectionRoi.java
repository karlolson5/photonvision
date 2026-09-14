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

import org.opencv.core.RotatedRect;

/**
 * A search ROI paired with where it came from, so the ML-Tag / Lookahead overlay can be
 * color-coded by source instead of drawing every box identically.
 *
 * @param rect the ROI itself, in full-resolution pixel coordinates
 * @param source whether this ROI came from a fresh NPU detection or from {@link TagRoiTracker}'s
 *     prediction
 */
public record DetectionRoi(RotatedRect rect, Source source) {
    public enum Source {
        /** Produced this frame by {@link AprilTagROIDetectionPipe} (the NPU model). */
        ML_DETECTED,
        /** Produced this frame by {@link TagRoiTracker#predict}, no NPU call made. */
        PREDICTED
    }

    /** Convenience factory for the common case of wrapping a fresh NPU detection ROI. */
    public static DetectionRoi mlDetected(RotatedRect rect) {
        return new DetectionRoi(rect, Source.ML_DETECTED);
    }

    /** Convenience factory for the common case of wrapping a lookahead-predicted ROI. */
    public static DetectionRoi predicted(RotatedRect rect) {
        return new DetectionRoi(rect, Source.PREDICTED);
    }
}
