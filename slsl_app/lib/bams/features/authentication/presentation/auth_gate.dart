import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import 'auth_controller.dart';
import 'login_page.dart';
import 'profile_page.dart';

class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  late final AuthController _controller;

  @override
  void initState() {
    super.initState();

    _controller = AuthController();
    _controller.initialise();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: AppTheme.light,
      child: ListenableBuilder(
        listenable: _controller,
        builder: (context, child) {
          switch (_controller.status) {
            case AuthStatus.checking:
              return Scaffold(
                backgroundColor: AppColors.background,
                body: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 64,
                        height: 64,
                        decoration: const BoxDecoration(
                          gradient: AppColors.primaryGradient,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.sign_language, color: Colors.white, size: 32),
                      ),
                      const SizedBox(height: 20),
                      const SizedBox(
                        width: 26,
                        height: 26,
                        child: CircularProgressIndicator(
                            strokeWidth: 2.5, color: AppColors.primary),
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'Restoring secure session...',
                        style: TextStyle(color: AppColors.inkSoft, fontSize: 13),
                      ),
                    ],
                  ),
                ),
              );

            case AuthStatus.authenticated:
              return ProfilePage(controller: _controller);

            case AuthStatus.unauthenticated:
              return LoginPage(controller: _controller);
          }
        },
      ),
    );
  }
}