// lib/services/teacher_api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../constants.dart';

class TeacherApiService {
  static Future<bool> checkHealth() async {
    try {
      final res = await http
          .get(Uri.parse('$kServerUrl/api/teacher/health'))
          .timeout(const Duration(seconds: 5));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<Map<String, dynamic>> validateFrame(String base64Image) async {
    try {
      final res = await http
          .post(
            Uri.parse('$kServerUrl/api/teacher/validate-frame'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'image': base64Image}),
          )
          .timeout(const Duration(seconds: 8));
      return jsonDecode(res.body);
    } catch (e) {
      return {
        'valid': false,
        'reason': 'connection_error',
        'message': 'Could not reach server. Check your connection.'
      };
    }
  }

  // Reuses Janith's existing /predict_frame endpoint (HTTP call only —
  // his server code is untouched) to get 63 keypoints from a single JPEG frame.
  static Future<List<double>> extractKeypoints(String base64Image, int frameId) async {
    try {
      final res = await http
          .post(
            Uri.parse('$kServerUrl/predict_frame'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'image': base64Image, 'frame_id': frameId}),
          )
          .timeout(const Duration(seconds: 8));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        return List<double>.from(data['keypoints']);
      }
    } catch (_) {}
    return List<double>.filled(kNumKeypoints, 0.0);
  }

  static Future<Map<String, dynamic>> submitSign({
    required String teacherId,
    required String englishWord,
    required String sinhalaWord,
    required String category,
    required List<List<double>> frames,
  }) async {
    try {
      final res = await http
          .post(
            Uri.parse('$kServerUrl/api/teacher/submit-sign'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'teacher_id': teacherId,
              'english_word': englishWord,
              'sinhala_word': sinhalaWord,
              'category': category,
              'frames': frames,
            }),
          )
          .timeout(const Duration(seconds: 20));
      final data = jsonDecode(res.body);
      data['_statusCode'] = res.statusCode;
      return data;
    } catch (e) {
      return {'error': 'Connection error: $e', '_statusCode': 0};
    }
  }

  static Future<List<Map<String, dynamic>>> getMySubmissions(String teacherId) async {
    try {
      final res = await http
          .get(Uri.parse('$kServerUrl/api/teacher/my-submissions/$teacherId'))
          .timeout(const Duration(seconds: 8));
      if (res.statusCode == 200) {
        return List<Map<String, dynamic>>.from(jsonDecode(res.body));
      }
    } catch (_) {}
    return [];
  }

  static Future<List<Map<String, dynamic>>> getVocabulary() async {
    try {
      final res = await http
          .get(Uri.parse('$kServerUrl/api/vocabulary'))
          .timeout(const Duration(seconds: 8));
      if (res.statusCode == 200) {
        return List<Map<String, dynamic>>.from(jsonDecode(res.body));
      }
    } catch (_) {}
    return [];
  }
}