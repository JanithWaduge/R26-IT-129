// lib/utils/sinhala_to_english.dart
//
// Sinhala word → English sign name lookup.
// English keys must exactly match sign names in assets/sign_data.json
// NEW file — does NOT touch any existing files.

const Map<String, String> sinhalaToEnglish = {
  // ── Demo set ──────────────────────────────────────────────────
  "ආයුබෝවන්":   "Hello",
  "පොත":        "Book",
  "පුටුව":      "Chair",
  "පරිගණකය":   "Computer",
  "පිළිතුර":    "Answer",

  // ── Full translations added here tomorrow ─────────────────────
};

/// Translate a Sinhala word to its English sign name.
/// Returns the original text if no translation found
/// (allows English words to pass through directly).
String translateToSign(String sinhalaText) {
  final trimmed = sinhalaText.trim();
  return sinhalaToEnglish[trimmed] ?? trimmed;
}
