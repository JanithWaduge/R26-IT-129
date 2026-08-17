// classifier_janith.dart
// ──────────────────────────────────────────────
// NOTE: ඔයාගේ app server-based inference use කරනවා.
// TFLite on-device inference use කරන්නේ නෑ.
// මේ file compatibility සඳහා keep කරලා තියෙනවා.
// Actual inference slsl_server.py (Flask) handle කරනවා.
// This function is NOT called anywhere in the current app (filtering
// happens server-side in slsl_server.py). Kept only as a reference /
// for a possible future on-device fallback — now rewritten so that IF
// it's ever used, it actually matches the server's real logic.
// ──────────────────────────────────────────────

import '../constants.dart';

// Noise filter — velocity threshold (research contribution)
// FIXED: now matches slsl_server.py's apply_noise_filter() exactly —
// velocity-filter, then drop no-hand frames, then sample down to
// kCaptureFrames (15) and zero-pad to kSequenceLength (30). The
// previous version truncated/padded without the zero-frame removal
// or the 15-frame sampling step, so it did NOT match the server.
List<List<double>> applyNoiseFilter(
  List<List<double>> sequence, {
  double threshold = 0.02,
  int captureFrames = 15, // must match slsl_server.py CAPTURE_FRAMES
}) {
  // Step 1 — velocity filter on the raw sequence
  List<List<double>> velocityFiltered;
  if (sequence.length < 2) {
    velocityFiltered = List<List<double>>.from(sequence);
  } else {
    velocityFiltered = <List<double>>[sequence[0]];
    for (int i = 1; i < sequence.length; i++) {
      double velocity = 0;
      for (int j = 0; j < sequence[i].length; j++) {
        velocity += (sequence[i][j] - sequence[i - 1][j]).abs();
      }
      velocity /= sequence[i].length;
      if (velocity > threshold) {
        velocityFiltered.add(sequence[i]);
      }
    }
  }

  // Step 2 — drop near-zero (no-hand) frames
  final valid = velocityFiltered
      .where((f) => f.fold<double>(0, (s, v) => s + v.abs()) > 0.01)
      .toList();

  if (valid.isEmpty) {
    return List.generate(
        kSequenceLength, (_) => List<double>.filled(kNumKeypoints, 0.0));
  }

  // Step 3 — sample down to captureFrames, then zero-pad to kSequenceLength
  final List<List<double>> sampled = [];
  if (valid.length >= captureFrames) {
    for (int i = 0; i < captureFrames; i++) {
      final idx = (i * (valid.length - 1) / (captureFrames - 1)).round();
      sampled.add(valid[idx]);
    }
  } else {
    for (int i = 0; i < captureFrames; i++) {
      final idx =
          (i * (valid.length - 1) / (captureFrames - 1)).round().clamp(0, valid.length - 1);
      sampled.add(valid[idx]);
    }
  }

  final padded = List<List<double>>.from(sampled);
  while (padded.length < kSequenceLength) {
    padded.add(List<double>.filled(kNumKeypoints, 0.0));
  }
  return padded;
}