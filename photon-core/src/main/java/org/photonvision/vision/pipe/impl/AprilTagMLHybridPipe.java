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
import java.util.List;
import org.opencv.core.RotatedRect;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;
import org.wpilib.vision.apriltag.AprilTagDetection;

/**
 * Composite pipe for ML-assisted AprilTag detection: runs YOLO ROI detection on the input frame,
 * pads each detected ROI, then runs the traditional WPILib AprilTag decoder on the matching regions
 * of the processed (single-channel) frame.
 *
 * <p>Each stage receives a different frame because each has different input requirements: the YOLO
 * model expects the full-fidelity source image matching its training distribution, while the WPILib
 * decoder requires a single-channel mat and performs its own adaptive thresholding internally.
 */
public class AprilTagMLHybridPipe
        extends CVPipe<Frame, MLDetectionResult, AprilTagMLHybridPipe.Params> implements Releasable {
    private final AprilTagROIDetectionPipe roiDetectionPipe = new AprilTagROIDetectionPipe();
    private final AprilTagROIDecodePipe roiDecodePipe = new AprilTagROIDecodePipe();

    public static class Params {
        public final AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams;
        public final AprilTagROIDecodePipe.ROIDecodeParams decodeParams;
        public final int roiPaddingPixels;

        public Params(
                AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams,
                AprilTagROIDecodePipe.ROIDecodeParams decodeParams,
                int roiPaddingPixels) {
            this.detectionParams = detectionParams;
            this.decodeParams = decodeParams;
            this.roiPaddingPixels = roiPaddingPixels;
        }
    }

    @Override
    public void setParams(Params newParams) {
        roiDetectionPipe.setParams(newParams.detectionParams);
        roiDecodePipe.setParams(newParams.decodeParams);
        super.setParams(newParams);
    }

    @Override
    protected MLDetectionResult process(Frame frame) {
        CVPipe.CVPipeResult<List<RotatedRect>> mlResult = roiDetectionPipe.run(frame.colorImage);
        List<RotatedRect> rawRois = mlResult.output;

        if (rawRois.isEmpty()) {
            return new MLDetectionResult(new ArrayList<>(), List.of());
        }

        int frameWidth = frame.colorImage.getMat().cols();
        int frameHeight = frame.colorImage.getMat().rows();
        List<RotatedRect> expandedRois = new ArrayList<>(rawRois.size());
        for (RotatedRect roi : rawRois) {
            expandedRois.add(
                    AprilTagROIDecodePipe.expandBbox(roi, params.roiPaddingPixels, frameWidth, frameHeight));
        }

        AprilTagROIDecodePipe.ROIDecodeInput decodeInput =
                new AprilTagROIDecodePipe.ROIDecodeInput(frame.processedImage, expandedRois);

        CVPipe.CVPipeResult<List<AprilTagDetection>> decodeResult = roiDecodePipe.run(decodeInput);

        return new MLDetectionResult(decodeResult.output, expandedRois);
    }

    /**
     * @return true when the ROI detection stage has a loaded model (not {@code NullModel}).
     */
    public boolean isAvailable() {
        return roiDetectionPipe.isAvailable();
    }

    /**
     * Runs only the decode stage against caller-supplied ROIs, skipping {@link
     * AprilTagROIDetectionPipe} (and therefore the NPU) entirely. Used by {@link TagRoiTracker} to
     * decode tags at predicted locations on frames where a full detection pass isn't needed.
     *
     * <p>Reuses this instance's already-configured {@code roiDecodePipe}, so decode parameters
     * (tag family, hamming distance, ATR settings, etc.) always match whatever the last {@link
     * #setParams} call configured for the full-detection path.
     *
     * @param frame the current frame; only {@code frame.processedImage} (single-channel) is used
     * @param rois predicted search regions, already expanded/padded by the caller
     * @return a result shaped like {@link #run}'s, so callers can handle both paths identically
     */
    public CVPipe.CVPipeResult<MLDetectionResult> decodeAt(Frame frame, List<RotatedRect> rois) {
        CVPipe.CVPipeResult<MLDetectionResult> wrapped = new CVPipe.CVPipeResult<>();

        if (rois.isEmpty()) {
            wrapped.output = new MLDetectionResult(new ArrayList<>(), List.of());
            wrapped.nanosElapsed = 0;
            return wrapped;
        }

        AprilTagROIDecodePipe.ROIDecodeInput decodeInput =
                new AprilTagROIDecodePipe.ROIDecodeInput(frame.processedImage, rois);
        CVPipe.CVPipeResult<List<AprilTagDetection>> decodeResult = roiDecodePipe.run(decodeInput);

        wrapped.output = new MLDetectionResult(decodeResult.output, rois);
        wrapped.nanosElapsed = decodeResult.nanosElapsed;
        return wrapped;
    }

    @Override
    public void release() {
        roiDetectionPipe.release();
        roiDecodePipe.release();
    }
}
