import 'package:flutter/material.dart';

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

    return Scaffold(
      appBar: AppBar(
        automaticallyImplyLeading: false,
        title: const Text('Quiz Result'),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            const Icon(Icons.emoji_events, size: 96),
            const SizedBox(height: 16),
            Text(
              '${score.round()}%',
              style: Theme.of(context).textTheme.displayMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text('Session ${session.status}', textAlign: TextAlign.center),
            const SizedBox(height: 32),
            Card(
              child: Column(
                children: [
                  _ResultTile(
                    label: 'Correct',
                    value: summary.correctCount,
                    icon: Icons.check_circle,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Incorrect',
                    value: summary.incorrectCount,
                    icon: Icons.cancel,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Skipped',
                    value: summary.skippedCount,
                    icon: Icons.skip_next,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Hints used',
                    value: summary.hintsUsed,
                    icon: Icons.lightbulb,
                  ),
                  const Divider(height: 1),
                  _ResultTile(
                    label: 'Retries',
                    value: summary.totalRetries,
                    icon: Icons.refresh,
                  ),
                ],
              ),
            ),

            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.psychology),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Your receptive or productive '
                        'mastery has been updated using '
                        'correctness, retries, hints, '
                        'response time and recognition '
                        'confidence where available.',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            if (reward != null) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      const Icon(Icons.auto_awesome, size: 48),
                      const SizedBox(height: 10),
                      Text(
                        '+${reward!.xpAwarded} XP',
                        style: Theme.of(context).textTheme.headlineMedium,
                      ),
                      Text('Level ${reward!.level}'),
                      const SizedBox(height: 12),
                      LinearProgressIndicator(
                        value: (reward!.xpIntoLevel / reward!.xpToNextLevel)
                            .clamp(0.0, 1.0),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${reward!.xpIntoLevel} / '
                        '${reward!.xpToNextLevel} XP',
                      ),
                      const SizedBox(height: 12),
                      Text('${reward!.currentStreakDays}-day learning streak'),
                    ],
                  ),
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
                        Text(
                          'Badges Unlocked',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 12),
                        ...reward!.badgesAwarded.map(
                          (GamificationBadge badge) => ListTile(
                            contentPadding: EdgeInsets.zero,
                            leading: const Icon(Icons.emoji_events),
                            title: Text(badge.title),
                            subtitle: Text(badge.description),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              const SizedBox(height: 16),
            ],
            const SizedBox(height: 24),
            FilledButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text('Finish'),
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
  });

  final String label;
  final int value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Icon(icon),
      title: Text(label),
      trailing: Text(
        value.toString(),
        style: Theme.of(context).textTheme.titleLarge,
      ),
    );
  }
}
