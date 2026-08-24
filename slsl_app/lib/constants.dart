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
// UI COLORS — white background, blue / violet / green accents
// (matches landing_screen.dart, home_screen_janith.dart, and the
// SLSL-BAMS module's app_colors.dart, so the whole app reads as one
// product instead of two different themes)
// ================================================
const Color kBackground = Color(0xFFFFFFFF); // was 0xFF03045E (navy) — now white
const Color kSurface    = Color(0xFFF6F8FC); // was 0xFF023E8A (blue) — now light card surface
const Color kPrimary    = Color(0xFF2F6BFF); // was 0xFF00B4D8 (cyan) — now Signal Blue
const Color kSecondary  = Color(0xFF8B5CF6); // was 0xFF7209B7 — now Violet, matches rest of app
const Color kAccent     = Color(0xFF8B5CF6); // kept for any file still referencing kAccent; mirrors kSecondary
const Color kSuccess    = Color(0xFF10B981); // was 0xFF06D6A0 — standardized green
const Color kWarning    = Color(0xFFF59E0B); // was 0xFFFFB703 — standardized amber
const Color kError      = Color(0xFFF43F5E); // was 0xFFEF233C — standardized coral

const Color kInk        = Color(0xFF14162B); // primary text on white — headings, titles
const Color kInkSoft    = Color(0xFF6B7280); // was 0xFFB8C4D9 (pale, made for dark bg —
                                              // invisible on white); now a proper muted gray

// ================================================
// RESEARCH-VALIDATED METRICS (offline, from noise_filter_experiment.py)
// Shown in the app as a credential ("this filter has been measured to
// do X"), NOT computed live. A true false-positive rate needs ground
// truth (real sign vs known-accidental movement), which a single live
// capture never has — only the offline experiment with synthetic
// accidental sequences can measure it honestly.
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