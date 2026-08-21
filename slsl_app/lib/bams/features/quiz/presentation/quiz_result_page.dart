import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import '../domain/quiz_session.dart';
import '../../gamification/domain/gamification_models.dart';

class QuizResultPage extends StatelessWidget {
  const QuizResultPage({
    required this.session,
    required this.reward,
    super.key,
  });

  final QuizSession session;
  final GamificationReward? reward;

  @override
  Widget build(BuildContext context) {
    final QuizSummary summary = session.summary;

    final int completed =
        summary.correctCount + summary.incorrectCount + summary.skippedCount;

    final double score = completed == 0
        ? 0
        : (summary.correctCount / completed) * 100;

    final Color scoreColor = score >= 80
        ? AppColors.success
        : score >= 50
            ? AppColors.warning
            : AppColors.error;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        automaticallyImplyLeading: false,
        title: const Text('Quiz Result'),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Center(
              child: Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: scoreColor.withOpacity(0.12),
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.emoji_events, size: 60, color: scoreColor),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              '${score.round()}%',
              style: TextStyle(color: scoreColor, fontSize: 40, fontWeight: FontWeight.w800),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text('Session ${session.status}',
                style: const TextStyle(color: AppColors.inkSoft),
                textAlign: TextAlign.center),
            const SizedBox(height: 32),
            Card(
              child: Column(
                children: [
                  _ResultTile(
                    label: 'Correct',
                    value: summary.correctCount,
                    icon: Icons.check_circle,
                    color: AppColors.success,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Incorrect',
                    value: summary.incorrectCount,
                    icon: Icons.cancel,
                    color: AppColors.error,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Skipped',
                    value: summary.skippedCount,
                    icon: Icons.skip_next,
                    color: AppColors.warning,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Hints used',
                    value: summary.hintsUsed,
                    icon: Icons.lightbulb,
                    color: AppColors.secondary,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Retries',
                    value: summary.totalRetries,
                    icon: Icons.refresh,
                    color: AppColors.primary,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.psychology, color: AppColors.indigo),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Your receptive or productive '
                        'mastery has been updated using '
                        'correctness, retries, hints, '
                        'response time and recognition '
                        'confidence where available.',
                        style: const TextStyle(color: AppColors.inkSoft, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            if (reward != null) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: AppColors.primaryGradient,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                        color: AppColors.primary.withOpacity(0.28),
                        blurRadius: 24,
                        offset: const Offset(0, 10)),
                  ],
                ),
                child: Column(
                  children: [
                    Container(
                      width: 60,
                      height: 60,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.18),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.auto_awesome, size: 30, color: Colors.white),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      '+${reward!.xpAwarded} XP',
                      style: const TextStyle(
                          color: Colors.white, fontSize: 24, fontWeight: FontWeight.w800),
                    ),
                    Text('Level ${reward!.level}',
                        style: const TextStyle(color: Colors.white70)),
                    const SizedBox(height: 12),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: LinearProgressIndicator(
                        value: (reward!.xpIntoLevel / reward!.xpToNextLevel)
                            .clamp(0.0, 1.0),
                        backgroundColor: Colors.white24,
                        valueColor: const AlwaysStoppedAnimation(Colors.white),
                        minHeight: 6,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${reward!.xpIntoLevel} / '
                      '${reward!.xpToNextLevel} XP',
                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                    const SizedBox(height: 12),
                    Text('${reward!.currentStreakDays}-day learning streak',
                        style: const TextStyle(color: Colors.white, fontSize: 13)),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              if (reward!.badgesAwarded.isNotEmpty)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Badges Unlocked',
                          style: TextStyle(
                              color: AppColors.ink, fontWeight: FontWeight.w700, fontSize: 15),
                        ),
                        const SizedBox(height: 12),
                        ...reward!.badgesAwarded.map(
                          (GamificationBadge badge) => ListTile(
                            contentPadding: EdgeInsets.zero,
                            leading: Container(
                              width: 38,
                              height: 38,
                              decoration: BoxDecoration(
                                color: AppColors.warning.withOpacity(0.14),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(Icons.emoji_events,
                                  color: AppColors.warning, size: 20),
                            ),
                            title: Text(badge.title,
                                style: const TextStyle(
                                    color: AppColors.ink, fontWeight: FontWeight.w600)),
                            subtitle: Text(badge.description,
                                style: const TextStyle(color: AppColors.inkSoft)),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              const SizedBox(height: 16),
            ],
            const SizedBox(height: 24),
            SizedBox(
              height: 52,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(14),
                  gradient: AppColors.primaryGradient,
                ),
                child: FilledButton(
                  style: FilledButton.styleFrom(
                      backgroundColor: Colors.transparent, shadowColor: Colors.transparent),
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: const Text('Finish'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ResultTile extends StatelessWidget {
  const _ResultTile({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });

  final String label;
  final int value;
  final IconData icon;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(color: color.withOpacity(0.14), shape: BoxShape.circle),
        child: Icon(icon, color: color, size: 18),
      ),
      title: Text(label, style: const TextStyle(color: AppColors.ink)),
      trailing: Text(
        value.toString(),
        style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.w800),
      ),
    );
  }
}