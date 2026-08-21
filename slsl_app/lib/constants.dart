import 'package:flutter/material.dart';

// ================================================
// SERVER — PC IP address change කරන්න
// ================================================
const String kServerUrl = 'http://172.20.10.5:5000';

// ================================================
// SIGN LABELS — 29 signs (Whiteboard Marker pending)
// ================================================
const List<String> kSignLabels = [
  'Allocate', 'Answer', 'Answer Properly', 'Answer Sheet', 'Ask Question',
  'Attend', 'Attending', 'Calculate', 'Cancel', 'Collaborating',
  'Collect', 'Comparing', 'Concentrate', 'Continuing', 'Coordinate',
  'Copying', 'Correct Mistake', 'Describe', 'Discuss', 'Discuss Topic',
  'Distribute', 'Documenting', 'Grade', 'Practice', 'Research',
  'Review', 'Study', 'Support', 'Teacher', 'Whiteboard Marker'
];

// ================================================
// MODEL SETTINGS
// ================================================
const int    kSequenceLength      = 30;
const int    kNumKeypoints        = 63;
const double kConfidenceThreshold = 0.60;

// ================================================
// UI COLORS
// ================================================


const Color kBackground = Color(0xFFFFFFFF);
const Color kSurface    = Color(0xFFF6F8FC);
const Color kInk        = Color(0xFF14162B);
const Color kInkSoft    = Color(0xFF6B7280);

const Color kPrimary    = Color(0xFF2F6BFF); // Signal Blue
const Color kSecondary  = Color(0xFF8B5CF6); // Violet
const Color kSuccess    = Color(0xFF10B981); // Green
const Color kWarning    = Color(0xFFF59E0B); // Amber (kept for pending/batch states)
const Color kError      = Color(0xFFF43F5E); // Coral (kept for delete/reject states)

