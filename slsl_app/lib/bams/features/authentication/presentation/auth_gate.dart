import 'package:flutter/material.dart';

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
    return ListenableBuilder(
      listenable: _controller,
      builder: (context, child) {
        switch (_controller.status) {
          case AuthStatus.checking:
            return const Scaffold(
              body: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Restoring secure session...'),
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
    );
  }
}
