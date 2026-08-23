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
const Color kPrimary    = Color(0xFF00B4D8);
const Color kAccent     = Color(0xFF90E0EF);
const Color kBackground = Color(0xFF03045E);
const Color kSurface    = Color(0xFF023E8A);
const Color kSuccess    = Color(0xFF06D6A0);
const Color kWarning    = Color(0xFFFFB703);
const Color kError      = Color(0xFFEF233C);

// ================================================
// ADDED — these were referenced in camera_screen_janith.dart
// (front-camera accent, muted status text) but were missing from
// this constants.dart, causing the kSecondary/kInkSoft errors.
// ⚠️ If your actual project already has these defined with specific
// values elsewhere (e.g. an AppColors/AppTheme file from the UI
// redesign), replace these two lines with your real values instead —
// these are reasonable placeholders that fit the existing palette,
// not a guaranteed match to your intended design.
// ================================================
const Color kSecondary  = Color(0xFF7209B7); // violet accent — front camera / secondary highlight
const Color kInkSoft    = Color(0xFFB8C4D9); // soft muted text, legible on kBackground
const Color kInk        = Color(0xFF1A1A2E); // strong/primary text — headings, titles, dialog text on light surfaces

// ================================================
// RESEARCH-VALIDATED METRICS (offline, from noise_filter_experiment.py)
// ADDED: shown in the app as a credential ("this filter has been
// measured to do X"), NOT computed live. A true false-positive rate
// needs ground truth (real sign vs known-accidental movement), which
// a single live capture never has — only the offline experiment with
// synthetic accidental sequences can measure it honestly.
//
// ⚠️ UPDATE THESE MANUALLY whenever you re-run
// noise_filter_experiment.py with a retrained model, so the app never
// displays stale numbers next to a newer model.
// ================================================
const double kResearchSignAccuracy      = 67.31; // Shared Model Accuracy (%)
const double kResearchFprBaseline       = 84.00; // Model A — FPR (%) on accidental data
const double kResearchFprFiltered       = 71.00; // Model B — FPR (%) on accidental data
const double kResearchFprReductionPct   = 15.48; // relative FPR reduction (%)
const double kResearchPValue            = 0.0003; // McNemar's test p-value
const String kResearchDatasetNote       = '258 samples · 30 signs · 80/20 split';