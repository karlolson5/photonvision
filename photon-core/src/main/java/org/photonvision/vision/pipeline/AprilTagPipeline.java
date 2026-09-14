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

package org.photonvision.vision.pipeline;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.opencv.core.RotatedRect;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.dataflow.structures.Packet;
import org.photonvision.common.hardware.Platform;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.estimation.TargetModel;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe.AprilTagDetectionPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDetectionPipe;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.DetectionRoi;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe.MultiTargetPNPPipeParams;
import org.photonvision.vision.pipe.impl.TagRoiTracker;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;
import org.photonvision.vision.target.TrackedTarget.TargetCalculationParameters;
import org.wpilib.math.filter.LinearFilter;
import org.wpilib.math.geometry.CoordinateSystem;
import org.wpilib.math.geometry.Pose3d;
import org.wpilib.math.geometry.Rotation3d;
import org.wpilib.math.geometry.Transform3d;
import org.wpilib.math.util.Units;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;
import org.wpilib.vision.apriltag.AprilTagPoseEstimate;
import org.wpilib.vision.apriltag.AprilTagPoseEstimator.Config;

public class AprilTagPipeline extends CVPipeline<CVPipelineResult, AprilTagPipelineSettings> {
    private static final Logger logger = new Logger(AprilTagPipeline.class, LogGroup.VisionModule);

    private final AprilTagDetectionPipe aprilTagDetectionPipe = new AprilTagDetectionPipe();
    private final AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe =
            new AprilTagPoseEstimatorPipe();
    private final MultiTargetPNPPipe multiTagPNPPipe = new MultiTargetPNPPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    // ML-assisted detection (composite ROI detect + ROI decode)
    private final AprilTagMLHybridPipe mlHybridPipe = new AprilTagMLHybridPipe();
    private boolean mlAvailable = false;
    private boolean mlWasAvailable = false;

    // Lookahead: predicts tag ROIs from recent frames so most frames can skip the NPU entirely.
    // One tracker per pipeline instance, i.e. per camera -- tracking state should not be shared
    // across cameras.
    private final TagRoiTracker roiTracker = new TagRoiTracker();

    // Frame dimensions the tracker's predictions are valid for. -1 forces a reset on the first
    // frame, which is harmless (the tracker already starts in a "must rescan" state) but keeps
    // the comparison below honest instead of accidentally matching a real 0x0 frame.
    private int lastFrameWidth = -1;
    private int lastFrameHeight = -1;

    // Running "lookahead hit rate" stat published on CVPipelineResult: a 30-frame moving average
    // of how often ML-detection frames used the lookahead (decode-only) path vs. a full NPU pass.
    // Uses the same LinearFilter.movingAverage the pipeline already uses for its FPS stat below,
    // rather than a lifetime cumulative average, so it actually reflects recent behavior instead
    // of slowly diluting toward a long-run mean over the course of a match.
    private final LinearFilter lookaheadHitRateFilter = LinearFilter.movingAverage(30);
    private double lookaheadHitRate = 0.0;

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.GREYSCALE;

    public AprilTagPipeline() {
        super(PROCESSING_TYPE);
        settings = new AprilTagPipelineSettings();
    }

    public AprilTagPipeline(AprilTagPipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {
        // Sanitize thread count - not supported to have fewer than 1 threads
        settings.threads = Math.max(1, settings.threads);

        // for now, hard code tag width based on enum value
        // From 2024 best guess is 6.5
        double tagWidth = Units.inchesToMeters(6.5);
        TargetModel tagModel = TargetModel.kAprilTag36h11;
        if (settings.tagFamily == AprilTagFamily.kTag16h5) {
            // 2023 tag, 6in
            tagWidth = Units.inchesToMeters(6);
            tagModel = TargetModel.kAprilTag16h5;
        }

        var config = new AprilTagDetector.Config();
        config.numThreads = settings.threads;
        config.refineEdges = settings.refineEdges;
        config.quadSigma = (float) settings.blur;
        config.quadDecimate = settings.decimate;

        var quadParams = new AprilTagDetector.QuadThresholdParameters();
        // 5 was the default minClusterPixels in WPILib prior to 2025
        // increasing it causes detection problems when decimate > 1
        quadParams.minClusterPixels = 5;
        // these are the same as the values in WPILib 2025
        // setting them here to prevent upstream changes from changing behavior of the detector
        quadParams.maxNumMaxima = 10;
        quadParams.criticalAngle = 45 * Math.PI / 180.0;
        quadParams.maxLineFitMSE = 10.0f;
        quadParams.minWhiteBlackDiff = 5;
        quadParams.deglitch = false;

        aprilTagDetectionPipe.setParams(
                new AprilTagDetectionPipeParams(settings.tagFamily, config, quadParams));

        if (frameStaticProperties.cameraCalibration != null) {
            var cameraMatrix = frameStaticProperties.cameraCalibration.getCameraIntrinsicsMat();
            if (cameraMatrix != null && cameraMatrix.rows() > 0) {
                var cx = cameraMatrix.get(0, 2)[0];
                var cy = cameraMatrix.get(1, 2)[0];
                var fx = cameraMatrix.get(0, 0)[0];
                var fy = cameraMatrix.get(1, 1)[0];

                singleTagPoseEstimatorPipe.setParams(
                        new AprilTagPoseEstimatorPipeParams(
                                new Config(tagWidth, fx, fy, cx, cy),
                                frameStaticProperties.cameraCalibration,
                                settings.numIterations));

                // TODO global state ew
                var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();
                multiTagPNPPipe.setParams(
                        new MultiTargetPNPPipeParams(frameStaticProperties.cameraCalibration, atfl, tagModel));
            }
        }

        // ML-assisted detection configuration
        if (settings.useMLDetection) {
            boolean platformOk = checkMLAvailability();
            Model apriltagModel = null;
            if (platformOk) {
                apriltagModel =
                        NeuralNetworkModelManager.getInstance()
                                .getModel(settings.model.modelPath().toString())
                                .orElse(null);
            }

            if (platformOk && apriltagModel != null) {
                AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams =
                        new AprilTagROIDetectionPipe.AprilTagROIDetectionParams(
                                apriltagModel, settings.mlConfidenceThreshold, settings.mlNmsThreshold);

                AprilTagROIDecodePipe.ROIDecodeParams decodeParams =
                        new AprilTagROIDecodePipe.ROIDecodeParams();
                decodeParams.tagFamily = settings.tagFamily;
                decodeParams.maxHammingDistance = settings.hammingDist;
                decodeParams.minDecisionMargin = settings.decisionMargin;
                decodeParams.detectorConfig.numThreads = settings.threads;
                decodeParams.detectorConfig.refineEdges = settings.refineEdges;
                decodeParams.detectorConfig.quadDecimate = settings.decimate;
                decodeParams.detectorConfig.quadSigma = (float) settings.blur;

                // ATR (Adaptive Tag Resizing) settings
                decodeParams.atrEnabled = settings.atrEnabled;
                decodeParams.atrTargetDimension = settings.atrTargetDimension;
                decodeParams.atrMinScaleFactor = settings.atrMinScaleFactor;

                mlHybridPipe.setParams(
                        new AprilTagMLHybridPipe.Params(
                                detectionParams, decodeParams, settings.mlRoiPaddingPixels));
            }

            mlAvailable = platformOk && apriltagModel != null && mlHybridPipe.isAvailable();

            if (!mlWasAvailable && mlAvailable) {
                logger.info("ML-assisted AprilTag detection enabled");
            }
            if (platformOk && apriltagModel == null && mlWasAvailable) {
                logger.warn("ML-assisted detection enabled but no AprilTag model found");
            }
            if (!platformOk && mlWasAvailable) {
                logger.debug("ML-assisted detection not available on this platform");
            }
        } else {
            mlAvailable = false;
        }
        if (!mlAvailable) {
            // No point tracking predicted ROIs for a detection path that isn't running.
            roiTracker.reset();
        }
        mlWasAvailable = mlAvailable;
    }

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagPipelineSettings settings) {
        long sumPipeNanosElapsed = 0L;

        if (frame.type != FrameThresholdType.GREYSCALE) {
            // We asked for a GREYSCALE frame, but didn't get one -- best we can do is give up
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }

        // Perform AprilTag detection (traditional or ML-assisted)
        List<AprilTagDetection> detections;
        long detectionNanos;
        List<DetectionRoi> mlDetectionRois = List.of();

        // Reset lookahead tracking if the frame size changed underneath us -- any predicted ROIs
        // we're holding are in stale pixel coordinates. See AprilTagPipeline#lastFrameWidth for
        // details.
        if (frame.processedImage.getMat().cols() != lastFrameWidth
                || frame.processedImage.getMat().rows() != lastFrameHeight) {
            roiTracker.reset();
            lastFrameWidth = frame.processedImage.getMat().cols();
            lastFrameHeight = frame.processedImage.getMat().rows();
        }

        if (settings.useMLDetection && mlAvailable) {
            if (settings.useLookahead
                    && roiTracker.shouldUseLookahead(
                            settings.lookaheadMaxCoastFrames, settings.lookaheadRescanIntervalFrames)) {
                // Tags are already being tracked confidently -- skip the NPU detection stage
                // entirely and just decode at the predicted locations.
                List<RotatedRect> predictedRois =
                        roiTracker.predict(
                                frame.processedImage.getMat().cols(),
                                frame.processedImage.getMat().rows(),
                                settings.lookaheadPaddingPixels);

                var lookaheadResult = mlHybridPipe.decodeAt(frame, predictedRois);
                detections = lookaheadResult.output.detections();
                detectionNanos = lookaheadResult.nanosElapsed;
                mlDetectionRois = predictedRois.stream().map(DetectionRoi::predicted).toList();

                roiTracker.updateFromLookahead(detections, settings.lookaheadMaxCoastFrames);
                lookaheadHitRate = lookaheadHitRateFilter.calculate(1.0);
            } else {
                // Cold start, tracking lapsed, or a periodic full rescan is due -- run the full
                // ML-assisted hybrid pass (NPU detection + decode).
                var hybridResult = mlHybridPipe.run(frame);
                detections = hybridResult.output.detections();
                detectionNanos = hybridResult.nanosElapsed;
                mlDetectionRois =
                        hybridResult.output.rois().stream().map(DetectionRoi::mlDetected).toList();

                if (settings.useLookahead) {
                    roiTracker.updateFromFullScan(detections);
                    lookaheadHitRate = lookaheadHitRateFilter.calculate(0.0);
                } else {
                    roiTracker.reset();
                    lookaheadHitRate = 0.0;
                }
            }
        } else {
            // Use traditional detection
            CVPipeResult<List<AprilTagDetection>> tagDetectionPipeResult =
                    aprilTagDetectionPipe.run(frame.processedImage);
            detections = tagDetectionPipeResult.output;
            detectionNanos = tagDetectionPipeResult.nanosElapsed;
            roiTracker.reset();
            lookaheadHitRate = 0.0;
        }
        sumPipeNanosElapsed += detectionNanos;

        List<AprilTagDetection> usedDetections = new ArrayList<>();
        List<TrackedTarget> targetList = new ArrayList<>();

        // Filter out detections based on pipeline settings
        for (AprilTagDetection detection : detections) {
            // TODO this should be in a pipe, not in the top level here (Matt)
            if (detection.getDecisionMargin() < settings.decisionMargin) continue;
            if (detection.getHamming() > settings.hammingDist) continue;

            usedDetections.add(detection);

            // Populate target list for multitag
            // (TODO: Address circular dependencies. Multitag only requires corners and IDs, this should
            // not be necessary.)
            TrackedTarget target =
                    new TrackedTarget(
                            detection,
                            null,
                            new TargetCalculationParameters(
                                    false, null, null, null, null, frameStaticProperties));

            targetList.add(target);
        }

        Optional<MultiTargetPNPResult> multiTagResult = Optional.empty();

        if (settings.solvePNPEnabled && settings.doMultiTarget) {
            var multiTagOutput = multiTagPNPPipe.run(targetList);
            sumPipeNanosElapsed += multiTagOutput.nanosElapsed;
            multiTagResult = multiTagOutput.output;
        }

        // Do single-tag pose estimation
        if (settings.solvePNPEnabled) {
            // Clear target list that was used for multitag so we can add target transforms
            targetList.clear();
            // TODO global state again ew
            var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();

            for (AprilTagDetection detection : usedDetections) {
                AprilTagPoseEstimate tagPoseEstimate = null;
                // Do single-tag estimation when "always enabled" or if a tag was not used for multitag
                if (settings.doSingleTargetAlways
                        || !(multiTagResult.isPresent()
                                && multiTagResult.get().fiducialIDsUsed.contains((short) detection.getId()))) {
                    var poseResult = singleTagPoseEstimatorPipe.run(detection);
                    sumPipeNanosElapsed += poseResult.nanosElapsed;
                    tagPoseEstimate = poseResult.output;
                }

                // If single-tag estimation was not done, this tag was used in multi-tag estimation
                if (tagPoseEstimate == null && multiTagResult.isPresent()) {
                    // compute this tag's camera-to-tag transform using the multitag result
                    var tagPose = atfl.getTagPose(detection.getId());
                    if (tagPose.isPresent()) {
                        var camToTag =
                                new Transform3d(
                                        new Pose3d().plus(multiTagResult.get().estimatedPose.best), tagPose.get());
                        // match expected AprilTag coordinate system
                        camToTag =
                                CoordinateSystem.convert(camToTag, CoordinateSystem.NWU(), CoordinateSystem.EDN());
                        // (AprilTag expects Z axis going into tag)
                        camToTag =
                                new Transform3d(
                                        camToTag.getTranslation(),
                                        new Rotation3d(0, Math.PI, 0).rotateBy(camToTag.getRotation()));
                        tagPoseEstimate = new AprilTagPoseEstimate(camToTag, camToTag, 0, 0);
                    }
                }

                // populate the target list
                // Challenge here is that TrackedTarget functions with OpenCV Contour
                TrackedTarget target =
                        new TrackedTarget(
                                detection,
                                tagPoseEstimate,
                                new TargetCalculationParameters(
                                        false, null, null, null, null, frameStaticProperties));

                var correctedBestPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getBestCameraToTarget3d());
                var correctedAltPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getAltCameraToTarget3d());

                target.setBestCameraToTarget3d(
                        new Transform3d(correctedBestPose.getTranslation(), correctedBestPose.getRotation()));
                target.setAltCameraToTarget3d(
                        new Transform3d(correctedAltPose.getTranslation(), correctedAltPose.getRotation()));

                targetList.add(target);
            }
        }

        if (targetList.size() > Packet.MAX_ARRAY_LEN) {
            logger.error(
                    "We have " + targetList.size() + " targets! Arbitrarily dropping some on the floor");
            targetList = targetList.subList(0, Packet.MAX_ARRAY_LEN);
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        var result =
                new CVPipelineResult(
                        frame.sequenceID,
                        sumPipeNanosElapsed,
                        fps,
                        targetList,
                        multiTagResult,
                        frame,
                        List.of(),
                        mlDetectionRois);

        result.lookaheadHitRate = lookaheadHitRate;

        return result;
    }

    /**
     * Checks if ML detection is available on the current platform. Currently supported: RK3588
     * (Orange Pi 5, Rock 5C, CoolPi 4B) and QCS6490 (Rubik Pi 3).
     */
    private boolean checkMLAvailability() {
        Platform platform = Platform.getCurrentPlatform();
        return platform == Platform.LINUX_QCS6490 || platform == Platform.LINUX_RK3588_64;
    }

    @Override
    public void release() {
        aprilTagDetectionPipe.release();
        singleTagPoseEstimatorPipe.release();
        mlHybridPipe.release();
        super.release();
    }
}
