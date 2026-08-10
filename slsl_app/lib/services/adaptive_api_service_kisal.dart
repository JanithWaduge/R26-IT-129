// lib/services/adaptive_api_service_kisal.dart
//
// Renamed from Kisal's original services/api_service.dart.
// Points at the SHARED Flask server (slsl_server.py) on the /adaptive
// prefix instead of a standalone FastAPI service on its own port.

import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/sign_item_kisal.dart';

class AdaptiveApiServiceKisal {
  // TODO: confirm this matches the host/port your teacher_api_service.dart
  // and classifier services already use for the shared Flask server.
  // 10.0.2.2 is the Android emulator's alias for the host machine's
  // localhost - keep that part if you're testing on an emulator.
  static const String baseUrl = "http://172.20.10.5:5000/adaptive";
  static const String currentUserId =
      "SE_STUDENT_01"; // Consistent placeholder session user

  String _readErrorMessage(http.Response response, String fallback) {
    try {
      final decoded = jsonDecode(response.body);
      if (decoded is Map<String, dynamic> && decoded['detail'] != null) {
        return decoded['detail'].toString();
      }
    } catch (_) {
      // Keep the original fallback when the server did not return JSON.
    }
    return fallback;
  }

  /// Pulls the prioritized/due items for a specific curriculum category
  Future<List<SignItemKisal>> fetchNextQuiz(
    String category, {
    bool includePractice = false,
  }) async {
    final url = Uri.parse(
      '$baseUrl/quiz/next?user_id=$currentUserId&category=$category&include_practice=$includePractice',
    );

    try {
      final response = await http.get(url);

      if (response.statusCode == 200) {
        List<dynamic> data = jsonDecode(response.body);
        return data.map((json) => SignItemKisal.fromJson(json)).toList();
      } else {
        final message = _readErrorMessage(
          response,
          "Failed to fetch adaptive quiz items from engine backend",
        );
        throw Exception("Server returned ${response.statusCode}: $message");
      }
    } catch (e) {
      if (e is Exception && e.toString().contains('Server returned')) {
        rethrow;
      }
      throw Exception("Network connectivity error: $e");
    }
  }

  /// Transmits the textual student response back down to the SM-2 calculator pipeline
  Future<Map<String, dynamic>> submitAnswer({
    required String signId,
    required String userAnswer,
    required double responseTimeSeconds,
  }) async {
    final url = Uri.parse('$baseUrl/quiz/submit');

    final payload = {
      "user_id": currentUserId,
      "sign_id": signId,
      "user_answer": userAnswer,
      "response_time_seconds": responseTimeSeconds,
    };

    try {
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        final message = _readErrorMessage(
          response,
          "Server failed to compute quiz telemetry metrics",
        );
        throw Exception("Server returned ${response.statusCode}: $message");
      }
    } catch (e) {
      if (e is Exception && e.toString().contains('Server returned')) {
        rethrow;
      }
      throw Exception("Network execution telemetry dropped: $e");
    }
  }

  /// Pulls the compiled metrics telemetry for the user dashboard
  Future<Map<String, dynamic>> fetchDashboardAnalytics() async {
    final url = Uri.parse(
      '$baseUrl/dashboard/analytics?user_id=$currentUserId',
    );

    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        final message = _readErrorMessage(
          response,
          "Failed to pull metrics from server",
        );
        throw Exception("Server returned ${response.statusCode}: $message");
      }
    } catch (e) {
      if (e is Exception && e.toString().contains('Server returned')) {
        rethrow;
      }
      throw Exception("Network analytics synchronization dropped: $e");
    }
  }
}