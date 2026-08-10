import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../../../core/config/api_config.dart';
import '../domain/health_status.dart';

class HealthService {
  HealthService({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  Future<HealthStatus> fetchHealth() async {
    final Uri uri = Uri.parse('${ApiConfig.baseUrl}/health');

    try {
      final http.Response response = await _client
          .get(uri, headers: const {'Accept': 'application/json'})
          .timeout(const Duration(seconds: 10));

      if (response.statusCode != 200) {
        throw HealthServiceException(
          'Server returned status code ${response.statusCode}.',
        );
      }

      final dynamic decodedBody = jsonDecode(response.body);

      if (decodedBody is! Map<String, dynamic>) {
        throw const HealthServiceException(
          'The server returned an invalid response.',
        );
      }

      return HealthStatus.fromJson(decodedBody);
    } on TimeoutException {
      throw const HealthServiceException('The backend connection timed out.');
    } on FormatException {
      throw const HealthServiceException('The backend returned invalid JSON.');
    } on http.ClientException catch (error) {
      throw HealthServiceException(
        'Network connection failed: ${error.message}',
      );
    }
  }

  void dispose() {
    _client.close();
  }
}

class HealthServiceException implements Exception {
  const HealthServiceException(this.message);

  final String message;

  @override
  String toString() => message;
}
