import 'dart:convert';

import 'package:http/http.dart' as http;

import '../../features/authentication/data/token_storage.dart';
import '../../features/authentication/domain/auth_tokens.dart';
import '../config/api_config.dart';

class AuthenticatedApiClient {
  AuthenticatedApiClient({http.Client? httpClient, TokenStorage? tokenStorage})
    : _httpClient = httpClient ?? http.Client(),
      _tokenStorage = tokenStorage ?? TokenStorage();

  final http.Client _httpClient;
  final TokenStorage _tokenStorage;

  Future<bool>? _refreshInProgress;

  Future<dynamic> getJson(
    String path, {
    Map<String, String>? queryParameters,
  }) async {
    Uri uri = Uri.parse('${ApiConfig.baseUrl}$path');

    if (queryParameters != null) {
      uri = uri.replace(queryParameters: queryParameters);
    }

    return _sendJson(method: 'GET', uri: uri);
  }

  Future<dynamic> postJson(String path, {Map<String, dynamic>? body}) async {
    final Uri uri = Uri.parse('${ApiConfig.baseUrl}$path');

    return _sendJson(method: 'POST', uri: uri, body: body);
  }

  Future<dynamic> _sendJson({
    required String method,
    required Uri uri,
    Map<String, dynamic>? body,
  }) async {
    http.Response response = await _sendAuthenticated(
      method: method,
      uri: uri,
      body: body,
    );

    if (response.statusCode == 401) {
      final bool refreshed = await _refreshSession();

      if (!refreshed) {
        await _tokenStorage.clearTokens();

        throw const ApiClientException(
          'Your session has expired. '
          'Please log in again.',
          statusCode: 401,
        );
      }

      response = await _sendAuthenticated(method: method, uri: uri, body: body);
    }

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw ApiClientException(
        _extractError(response),
        statusCode: response.statusCode,
      );
    }

    if (response.body.isEmpty) {
      return null;
    }

    try {
      return jsonDecode(response.body);
    } on FormatException {
      throw const ApiClientException('The server returned invalid JSON.');
    }
  }

  Future<http.Response> _sendAuthenticated({
    required String method,
    required Uri uri,
    Map<String, dynamic>? body,
  }) async {
    final String? accessToken = await _tokenStorage.readAccessToken();

    if (accessToken == null) {
      throw const ApiClientException(
        'No active login session exists.',
        statusCode: 401,
      );
    }

    final Map<String, String> headers = {
      'Accept': 'application/json',
      'Authorization': 'Bearer $accessToken',
    };

    if (method == 'POST') {
      headers['Content-Type'] = 'application/json';

      return _httpClient.post(
        uri,
        headers: headers,
        body: jsonEncode(body ?? <String, dynamic>{}),
      );
    }

    return _httpClient.get(uri, headers: headers);
  }

  Future<bool> _refreshSession() {
    final Future<bool>? existing = _refreshInProgress;

    if (existing != null) {
      return existing;
    }

    final Future<bool> refreshFuture = _performRefresh();

    _refreshInProgress = refreshFuture;

    return refreshFuture.whenComplete(() {
      _refreshInProgress = null;
    });
  }

  Future<bool> _performRefresh() async {
    final String? refreshToken = await _tokenStorage.readRefreshToken();

    if (refreshToken == null) {
      return false;
    }

    final http.Response response = await _httpClient.post(
      Uri.parse(
        '${ApiConfig.baseUrl}'
        '/auth/refresh',
      ),
      headers: const {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({'refresh_token': refreshToken}),
    );

    if (response.statusCode != 200) {
      return false;
    }

    try {
      final dynamic decoded = jsonDecode(response.body);

      if (decoded is! Map<String, dynamic>) {
        return false;
      }

      final AuthTokens tokens = AuthTokens.fromJson(decoded);

      await _tokenStorage.saveTokens(tokens);

      return true;
    } catch (_) {
      return false;
    }
  }

  String _extractError(http.Response response) {
    try {
      final dynamic decoded = jsonDecode(response.body);

      if (decoded is Map<String, dynamic>) {
        final dynamic detail = decoded['detail'];

        if (detail is String) {
          return detail;
        }
      }
    } catch (_) {
      // Use a generic error.
    }

    return 'Request failed with status '
        '${response.statusCode}.';
  }

  void dispose() {
    _httpClient.close();
  }
}

class ApiClientException implements Exception {
  const ApiClientException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}
