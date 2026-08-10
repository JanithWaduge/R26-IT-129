import 'dart:convert';

import 'package:http/http.dart' as http;

import '../../../core/config/api_config.dart';
import '../domain/auth_tokens.dart';
import '../domain/user_profile.dart';
import 'token_storage.dart';

class AuthApi {
  AuthApi({http.Client? client, TokenStorage? tokenStorage})
    : _client = client ?? http.Client(),
      _tokenStorage = tokenStorage ?? TokenStorage();

  final http.Client _client;
  final TokenStorage _tokenStorage;

  Future<void> register({
    required String fullName,
    required String email,
    required String password,
    required String preferredLanguage,
    required String gradeLevel,
  }) async {
    final http.Response response = await _client.post(
      Uri.parse('${ApiConfig.baseUrl}/auth/register'),
      headers: const {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'full_name': fullName.trim(),
        'email': email.trim().toLowerCase(),
        'password': password,
        'preferred_language': preferredLanguage,
        'grade_level': gradeLevel.trim(),
      }),
    );

    if (response.statusCode != 201) {
      throw AuthApiException(_extractError(response));
    }

    final AuthTokens tokens = AuthTokens.fromJson(_decodeObject(response.body));

    await _tokenStorage.saveTokens(tokens);
  }

  Future<void> login({required String email, required String password}) async {
    final http.Response response = await _client.post(
      Uri.parse('${ApiConfig.baseUrl}/auth/login'),
      headers: const {
        'Accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: {'username': email.trim().toLowerCase(), 'password': password},
    );

    if (response.statusCode != 200) {
      throw AuthApiException(_extractError(response));
    }

    final AuthTokens tokens = AuthTokens.fromJson(_decodeObject(response.body));

    await _tokenStorage.saveTokens(tokens);
  }

  Future<UserProfile> fetchProfile() async {
    final String? accessToken = await _tokenStorage.readAccessToken();

    if (accessToken == null) {
      throw const AuthApiException('No active session was found.');
    }

    http.Response response = await _getProfile(accessToken);

    if (response.statusCode == 401) {
      final bool refreshed = await _refreshSession();

      if (!refreshed) {
        await _tokenStorage.clearTokens();

        throw const AuthApiException(
          'Your session has expired. '
          'Please log in again.',
        );
      }

      final String? renewedAccessToken = await _tokenStorage.readAccessToken();

      if (renewedAccessToken == null) {
        throw const AuthApiException('The renewed session is unavailable.');
      }

      response = await _getProfile(renewedAccessToken);
    }

    if (response.statusCode != 200) {
      throw AuthApiException(_extractError(response));
    }

    return UserProfile.fromJson(_decodeObject(response.body));
  }

  Future<bool> restoreSession() async {
    final bool hasRefreshToken = await _tokenStorage.hasRefreshToken();

    if (!hasRefreshToken) {
      return false;
    }

    try {
      await fetchProfile();
      return true;
    } on AuthApiException {
      await _tokenStorage.clearTokens();
      return false;
    }
  }

  Future<void> logout() async {
    final String? refreshToken = await _tokenStorage.readRefreshToken();

    try {
      if (refreshToken != null) {
        await _client.post(
          Uri.parse('${ApiConfig.baseUrl}/auth/logout'),
          headers: const {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
          body: jsonEncode({'refresh_token': refreshToken}),
        );
      }
    } finally {
      await _tokenStorage.clearTokens();
    }
  }

  Future<void> logoutAll() async {
    final String? accessToken = await _tokenStorage.readAccessToken();

    try {
      if (accessToken != null) {
        await _client.post(
          Uri.parse(
            '${ApiConfig.baseUrl}'
            '/auth/logout-all',
          ),
          headers: {
            'Accept': 'application/json',
            'Authorization': 'Bearer $accessToken',
          },
        );
      }
    } finally {
      await _tokenStorage.clearTokens();
    }
  }

  Future<http.Response> _getProfile(String accessToken) {
    return _client.get(
      Uri.parse('${ApiConfig.baseUrl}/auth/me'),
      headers: {
        'Accept': 'application/json',
        'Authorization': 'Bearer $accessToken',
      },
    );
  }

  Future<bool> _refreshSession() async {
    final String? refreshToken = await _tokenStorage.readRefreshToken();

    if (refreshToken == null) {
      return false;
    }

    final http.Response response = await _client.post(
      Uri.parse('${ApiConfig.baseUrl}/auth/refresh'),
      headers: const {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({'refresh_token': refreshToken}),
    );

    if (response.statusCode != 200) {
      return false;
    }

    final AuthTokens tokens = AuthTokens.fromJson(_decodeObject(response.body));

    await _tokenStorage.saveTokens(tokens);

    return true;
  }

  Map<String, dynamic> _decodeObject(String responseBody) {
    final dynamic decoded = jsonDecode(responseBody);

    if (decoded is! Map<String, dynamic>) {
      throw const AuthApiException('The server returned an invalid response.');
    }

    return decoded;
  }

  String _extractError(http.Response response) {
    try {
      final dynamic decoded = jsonDecode(response.body);

      if (decoded is Map<String, dynamic>) {
        final dynamic detail = decoded['detail'];

        if (detail is String) {
          return detail;
        }

        if (detail is List && detail.isNotEmpty) {
          final dynamic first = detail.first;

          if (first is Map<String, dynamic>) {
            return first['msg'] as String? ?? 'Validation failed.';
          }
        }
      }
    } catch (_) {
      // Fall through to generic message.
    }

    return 'Request failed with status '
        '${response.statusCode}.';
  }

  void dispose() {
    _client.close();
  }
}

class AuthApiException implements Exception {
  const AuthApiException(this.message);

  final String message;

  @override
  String toString() => message;
}
