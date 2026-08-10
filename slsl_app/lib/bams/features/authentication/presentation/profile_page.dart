import 'package:flutter/material.dart';
import '../../vocabulary/presentation/vocabulary_page.dart';
import '../../mastery/presentation/mastery_page.dart';
import '../../review/presentation/review_page.dart';
import '../../quiz/presentation/quiz_setup_page.dart';
import '../../gamification/presentation/learning_dashboard_page.dart';

import 'auth_controller.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({required this.controller, super.key});

  final AuthController controller;

  @override
  Widget build(BuildContext context) {
    final profile = controller.profile;

    if (profile == null) {
      return const Scaffold(
        body: Center(child: Text('Profile is unavailable.')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Student Profile'),
        actions: [
          PopupMenuButton<String>(
            onSelected: (String selection) async {
              if (selection == 'logout') {
                await controller.logout();
              }

              if (selection == 'logout_all') {
                await controller.logoutAll();
              }
            },
            itemBuilder: (context) {
              return const [
                PopupMenuItem(value: 'logout', child: Text('Log out')),
                PopupMenuItem(
                  value: 'logout_all',
                  child: Text('Log out from all devices'),
                ),
              ];
            },
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            const Icon(Icons.account_circle, size: 100),
            const SizedBox(height: 16),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (context) {
                      return VocabularyPage(
                        preferredLanguage: profile.preferredLanguage,
                      );
                    },
                  ),
                );
              },
              icon: const Icon(Icons.sign_language),
              label: const Text('Open Vocabulary'),
            ),
            Text(
              profile.fullName,
              style: Theme.of(context).textTheme.headlineMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(profile.email, textAlign: TextAlign.center),
            const SizedBox(height: 32),
            Card(
              child: Column(
                children: [
                  ListTile(
                    leading: const Icon(Icons.badge),
                    title: const Text('Role'),
                    subtitle: Text(profile.role),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.language),
                    title: const Text('Preferred language'),
                    subtitle: Text(profile.preferredLanguage),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.school),
                    title: const Text('Level'),
                    subtitle: Text(profile.gradeLevel),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (context) {
                      return QuizSetupPage(
                        preferredLanguage: profile.preferredLanguage,
                      );
                    },
                  ),
                );
              },
              icon: const Icon(Icons.quiz),
              label: const Text('Start Adaptive Quiz'),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (context) {
                      return MasteryPage(
                        preferredLanguage: profile.preferredLanguage,
                      );
                    },
                  ),
                );
              },
              icon: const Icon(Icons.insights),
              label: const Text('View My Progress'),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (context) => ReviewPage(
                      preferredLanguage: profile.preferredLanguage,
                    ),
                  ),
                );
              },
              icon: const Icon(Icons.schedule),
              label: const Text('Review Due Signs'),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (context) => const LearningDashboardPage(),
                  ),
                );
              },
              icon: const Icon(Icons.dashboard),
              label: const Text('Learning Dashboard'),
            ),
            const SizedBox(height: 24),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    const Icon(Icons.lock, size: 42),
                    const SizedBox(height: 12),
                    Text(
                      'Secure session active',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Your access token is '
                      'used for protected API '
                      'requests. The refresh '
                      'token is stored in '
                      'protected device storage.',
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
