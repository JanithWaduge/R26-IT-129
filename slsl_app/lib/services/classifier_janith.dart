// classifier_janith.dart
// ──────────────────────────────────────────────
// NOTE: ඔයාගේ app server-based inference use කරනවා.
// TFLite on-device inference use කරන්නේ නෑ.
// මේ file compatibility සඳහා keep කරලා තියෙනවා.
// Actual inference slsl_server.py (Flask) handle කරනවා.
// ──────────────────────────────────────────────

import '../constants.dart';

// Noise filter — velocity threshold (research contribution)
// Server side apply_noise_filter() ටම match කරන client-side version
List<List<double>> applyNoiseFilter(
  List<List<double>> sequence, {
  double threshold = 0.02,
}) {
  if (sequence.length < 2) return sequence;

  final filtered = <List<double>>[sequence[0]];

  for (int i = 1; i < sequence.length; i++) {
    double velocity = 0;
    for (int j = 0; j < sequence[i].length; j++) {
      velocity += (sequence[i][j] - sequence[i - 1][j]).abs();
    }
    velocity /= sequence[i].length;

    if (velocity > threshold) {
      filtered.add(sequence[i]);
    }
  }

  if (filtered.length >= kSequenceLength) {
    return filtered.sublist(0, kSequenceLength);
  } else {
    final padded = List<List<double>>.from(filtered);
    while (padded.length < kSequenceLength) {
      padded.add(List<double>.filled(kNumKeypoints, 0.0));
    }
    return padded;
  }
}