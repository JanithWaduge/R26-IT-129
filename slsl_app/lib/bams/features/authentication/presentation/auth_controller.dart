import 'package:flutter/foundation.dart';

import '../data/auth_api.dart';
import '../domain/user_profile.dart';

enum AuthStatus { checking, authenticated, unauthenticated }

class AuthController extends ChangeNotifier {
  AuthController({AuthApi? authApi}) : _authApi = authApi ?? AuthApi();

  final AuthApi _authApi;

  AuthStatus _status = AuthStatus.checking;
  UserProfile? _profile;
  String? _errorMessage;
  bool _isSubmitting = false;

  AuthStatus get status => _status;
  UserProfile? get profile => _profile;
  String? get errorMessage => _errorMessage;
  bool get isSubmitting => _isSubmitting;

  Future<void> initialise() async {
    _status = AuthStatus.checking;
    notifyListeners();

    final bool restored = await _authApi.restoreSession();

    if (!restored) {
      _profile = null;
      _status = AuthStatus.unauthenticated;
      notifyListeners();
      return;
    }

    try {
      _profile = await _authApi.fetchProfile();

      _status = AuthStatus.authenticated;
    } catch (_) {
      _profile = null;
      _status = AuthStatus.unauthenticated;
    }

    notifyListeners();
  }

  Future<bool> login({required String email, required String password}) async {
    return _performAuthentication(
      action: () => _authApi.login(email: email, password: password),
    );
  }

  Future<bool> register({
    required String fullName,
    required String email,
    required String password,
    required String preferredLanguage,
    required String gradeLevel,
  }) async {
    return _performAuthentication(
      action: () => _authApi.register(
        fullName: fullName,
        email: email,
        password: password,
        preferredLanguage: preferredLanguage,
        gradeLevel: gradeLevel,
      ),
    );
  }

  Future<bool> _performAuthentication({
    required Future<void> Function() action,
  }) async {
    _isSubmitting = true;
    _errorMessage = null;
    notifyListeners();

    try {
      await action();

      _profile = await _authApi.fetchProfile();

      _status = AuthStatus.authenticated;

      return true;
    } on AuthApiException catch (error) {
      _errorMessage = error.message;
      _status = AuthStatus.unauthenticated;

      return false;
    } catch (_) {
      _errorMessage = 'An unexpected error occurred.';
      _status = AuthStatus.unauthenticated;

      return false;
    } finally {
      _isSubmitting = false;
      notifyListeners();
    }
  }

  Future<void> logout() async {
    await _authApi.logout();

    _profile = null;
    _errorMessage = null;
    _status = AuthStatus.unauthenticated;

    notifyListeners();
  }

  Future<void> logoutAll() async {
    await _authApi.logoutAll();

    _profile = null;
    _errorMessage = null;
    _status = AuthStatus.unauthenticated;

    notifyListeners();
  }

  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _authApi.dispose();
    super.dispose();
  }
}
