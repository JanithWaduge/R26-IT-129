import 'package:flutter/material.dart';
import '../../../core/theme/app_colors.dart';
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
        backgroundColor: AppColors.background,
        body: Center(
            child: Text('Profile is unavailable.',
                style: TextStyle(color: AppColors.inkSoft))),
      );
    }

    return Scaffold(
      backgroundColor: AppColors.background,
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
          padding: const EdgeInsets.all(20),
          children: [
            // ── Header: gradient identity card ──
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(22),
              decoration: BoxDecoration(
                gradient: AppColors.primaryGradient,
                borderRadius: BorderRadius.circular(22),
                boxShadow: [
                  BoxShadow(
                      color: AppColors.primary.withOpacity(0.28),
                      blurRadius: 26,
                      offset: const Offset(0, 12)),
                ],
              ),
              child: Column(
                children: [
                  Container(
                    width: 76,
                    height: 76,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.18),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.account_circle,
                        size: 60, color: Colors.white),
                  ),
                  const SizedBox(height: 14),
                  Text(
                    profile.fullName,
                    style: const TextStyle(
                        color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 4),
                  Text(profile.email,
                      style: const TextStyle(color: Colors.white70, fontSize: 13),
                      textAlign: TextAlign.center),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Card(
              child: Column(
                children: [
                  ListTile(
                    leading: const Icon(Icons.badge, color: AppColors.primary),
                    title: const Text('Role'),
                    subtitle: Text(profile.role),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.language, color: AppColors.success),
                    title: const Text('Preferred language'),
                    subtitle: Text(profile.preferredLanguage),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.school, color: AppColors.secondary),
                    title: const Text('Level'),
                    subtitle: Text(profile.gradeLevel),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            const Text('QUICK ACTIONS',
                style: TextStyle(
                    color: AppColors.inkSoft,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5)),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(
                child: _ActionTile(
                  icon: Icons.sign_language,
                  label: 'Open Vocabulary',
                  color: AppColors.secondary,
                  onTap: () {
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
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _ActionTile(
                  icon: Icons.quiz,
                  label: 'Start Adaptive Quiz',
                  color: AppColors.primary,
                  onTap: () {
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
                ),
              ),
            ]),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(
                child: _ActionTile(
                  icon: Icons.insights,
                  label: 'View My Progress',
                  color: AppColors.success,
                  onTap: () {
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
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _ActionTile(
                  icon: Icons.schedule,
                  label: 'Review Due Signs',
                  color: AppColors.warning,
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (context) => ReviewPage(
                          preferredLanguage: profile.preferredLanguage,
                        ),
                      ),
                    );
                  },
                ),
              ),
            ]),
            const SizedBox(height: 12),
            _ActionTile(
              icon: Icons.dashboard,
              label: 'Learning Dashboard',
              color: AppColors.indigo,
              fullWidth: true,
              onTap: () {
                Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (context) => const LearningDashboardPage(),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    Container(
                      width: 56,
                      height: 56,
                      decoration: BoxDecoration(
                        color: AppColors.success.withOpacity(0.12),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.lock, size: 28, color: AppColors.success),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Secure session active',
                      style: TextStyle(
                          color: AppColors.ink, fontSize: 15, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Your access token is '
                      'used for protected API '
                      'requests. The refresh '
                      'token is stored in '
                      'protected device storage.',
                      style: TextStyle(color: AppColors.inkSoft, fontSize: 12, height: 1.5),
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

class _ActionTile extends StatelessWidget {
  const _ActionTile({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
    this.fullWidth = false,
  });

  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;
  final bool fullWidth;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: fullWidth ? double.infinity : null,
        padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.hairline),
          boxShadow: [
            BoxShadow(color: color.withOpacity(0.14), blurRadius: 16, offset: const Offset(0, 8)),
          ],
        ),
        child: fullWidth
            ? Row(children: [
                _iconBadge(),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(label,
                      style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 14)),
                ),
                Icon(Icons.arrow_forward_ios_rounded, color: color.withOpacity(0.6), size: 14),
              ])
            : Column(children: [
                _iconBadge(),
                const SizedBox(height: 10),
                Text(label,
                    textAlign: TextAlign.center,
                    style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 13)),
              ]),
      ),
    );
  }

  Widget _iconBadge() => Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          gradient: LinearGradient(colors: [color, color.withOpacity(0.7)]),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: Colors.white, size: 20),
      );
}