import 'package:flutter/material.dart';

// ================================================
// SERVER
// ================================================
const String kServerUrl = 'http://172.20.10.5:5000';  // ← PC IP address

// ================================================
// SIGN LABELS
// ================================================
const List<String> kSignLabels = [
  'Allocate', 'Answer', 'Answer Properly', 'Answer Sheet', 'Ask Question',
  'Attend', 'Attending', 'Calculate', 'Cancel', 'Collaborating',
  'Collect', 'Comparing', 'Concentrate', 'Continuing', 'Coordinate',
  'Copying', 'Correct Mistake', 'Describe', 'Discuss', 'Discuss Topic',
  'Distribute', 'Documenting', 'Grade', 'Practice', 'Research',
  'Review', 'Study', 'Support', 'Teacher', 'Whiteboard Marker',
];

// ================================================
// MODEL SETTINGS
// ================================================
const String kModelPath           = 'assets/slsl_model.tflite';
const int    kSequenceLength      = 30;
const int    kNumKeypoints        = 63;
const double kConfidenceThreshold = 0.60;

// ================================================
// UI COLORS
// ================================================
const Color kPrimary    = Color(0xFF00B4D8);
const Color kAccent     = Color(0xFF90E0EF);
const Color kBackground = Color(0xFF03045E);
const Color kSurface    = Color(0xFF023E8A);
const Color kSuccess    = Color(0xFF06D6A0);
const Color kWarning    = Color(0xFFFFB703);
const Color kError      = Color(0xFFEF233C);