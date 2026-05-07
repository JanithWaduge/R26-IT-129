import 'package:tflite_flutter/tflite_flutter.dart';
import '../constants.dart';

class SignClassifier {
  Interpreter? _interpreter;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  // ── Load model ───────────────────────────────
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(kModelPath);
      _isLoaded = true;
      print('✅ TFLite model loaded successfully');
      print('   Input shape : ${_interpreter!.getInputTensor(0).shape}');
      print('   Output shape: ${_interpreter!.getOutputTensor(0).shape}');
    } catch (e) {
      _isLoaded = false;
      print('❌ Failed to load model: $e');
    }
  }

  // ── Run inference ────────────────────────────
  // sequence: List of 30 frames, each frame = 63 floats
  ClassificationResult? classify(List<List<double>> sequence) {
    if (!_isLoaded || _interpreter == null) return null;
    if (sequence.length != kSequenceLength) return null;

    try {
      // Input: shape [1, 30, 63]
      final input = [sequence];

      // Output: shape [1, num_classes] — reshape ගේ වෙනුවට List directly
      final output = [List<double>.filled(kSignLabels.length, 0.0)];

      _interpreter!.run(input, output);

      final scores = List<double>.from(output[0]);

      // Get top result
      double maxScore = 0;
      int maxIndex = 0;
      for (int i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
          maxScore = scores[i];
          maxIndex = i;
        }
      }

      // Get top 3 results
      final indexed = scores
          .asMap()
          .entries
          .map((e) => MapEntry(e.key, e.value))
          .toList()
        ..sort((a, b) => b.value.compareTo(a.value));

      final top3 = indexed.take(3).map((e) {
        return TopResult(
          label     : kSignLabels[e.key],
          confidence: e.value,
        );
      }).toList();

      return ClassificationResult(
        label     : kSignLabels[maxIndex],
        confidence: maxScore,
        top3      : top3,
      );
    } catch (e) {
      print('❌ Inference error: $e');
      return null;
    }
  }

  // ── Apply noise filter ───────────────────────
  // Velocity threshold filter (research contribution)
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

    // Normalize to kSequenceLength
    if (filtered.length >= kSequenceLength) {
      return filtered.sublist(0, kSequenceLength);
    } else {
      // Zero padding
      final padded = List<List<double>>.from(filtered);
      while (padded.length < kSequenceLength) {
        padded.add(List<double>.filled(kNumKeypoints, 0.0));
      }
      return padded;
    }
  }

  void dispose() {
    _interpreter?.close();
    _isLoaded = false;
  }
}

// ── Data models ──────────────────────────────

class ClassificationResult {
  final String         label;
  final double         confidence;
  final List<TopResult> top3;

  ClassificationResult({
    required this.label,
    required this.confidence,
    required this.top3,
  });
}

class TopResult {
  final String label;
  final double confidence;

  TopResult({
    required this.label,
    required this.confidence,
  });
}