import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

// ── Server config ─────────────────────────────────────────────────────────
// Change this to your PC's LAN IP address before running on a physical device.
// Find it with: ipconfig (Windows) → "IPv4 Address" of your Wi-Fi adapter.
// Example: 'http://192.168.1.42:8000'
// Android emulator uses 10.0.2.2 to reach the host machine.
const String kAsrServerUrl = 'http://10.79.21.207:8000';

// ── Data model ────────────────────────────────────────────────────────────

/// Holds the result returned by the Whisper ASR server.
class AsrResult {
  final String text;
  final String language;      // ISO 639-1 code, e.g. "si"
  final String languageName;  // Display name, e.g. "Sinhala"
  final String locale;        // BCP-47 tag, e.g. "si-LK"
  final double confidence;    // 0.0 – 1.0

  const AsrResult({
    required this.text,
    required this.language,
    required this.languageName,
    required this.locale,
    required this.confidence,
  });

  factory AsrResult.fromJson(Map<String, dynamic> json) => AsrResult(
        text:         json['text']          as String,
        language:     json['language']      as String,
        languageName: json['language_name'] as String,
        locale:       json['locale']        as String,
        confidence:   (json['confidence']   as num).toDouble(),
      );
}

// ── Service ───────────────────────────────────────────────────────────────

class AsrService {
  AsrService._();

  /// Send [audioFile] to the ASR server and return transcription.
  ///
  /// [language] is optional — pass "si" or "ta" to force a language,
  /// or leave null for Whisper to auto-detect.
  static Future<AsrResult> transcribe(
    File audioFile, {
    String? language,
  }) async {
    final uri = Uri.parse('$kAsrServerUrl/asr/transcribe');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath('file', audioFile.path),
    );
    if (language != null) {
      request.fields['language'] = language;
    }

    final streamed = await request.send().timeout(const Duration(seconds: 120));
    final body     = await streamed.stream.bytesToString();

    if (streamed.statusCode != 200) {
      throw Exception('ASR server error ${streamed.statusCode}: $body');
    }

    return AsrResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  /// Returns true if the ASR server is reachable.
  /// Flutter calls this on screen load to show a warning if the PC is offline.
  static Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$kAsrServerUrl/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
