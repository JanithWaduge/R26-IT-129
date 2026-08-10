import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../domain/auth_tokens.dart';

class TokenStorage {
  TokenStorage({FlutterSecureStorage? storage})
    : _storage =
          storage ?? const FlutterSecureStorage(aOptions: AndroidOptions());

  static const String _accessTokenKey = 'slsl_bams_access_token';

  static const String _refreshTokenKey = 'slsl_bams_refresh_token';

  final FlutterSecureStorage _storage;

  Future<void> saveTokens(AuthTokens tokens) async {
    await Future.wait([
      _storage.write(key: _accessTokenKey, value: tokens.accessToken),
      _storage.write(key: _refreshTokenKey, value: tokens.refreshToken),
    ]);
  }

  Future<String?> readAccessToken() {
    return _storage.read(key: _accessTokenKey);
  }

  Future<String?> readRefreshToken() {
    return _storage.read(key: _refreshTokenKey);
  }

  Future<bool> hasRefreshToken() async {
    final String? token = await readRefreshToken();

    return token != null && token.isNotEmpty;
  }

  Future<void> clearTokens() async {
    await Future.wait([
      _storage.delete(key: _accessTokenKey),
      _storage.delete(key: _refreshTokenKey),
    ]);
  }
}
