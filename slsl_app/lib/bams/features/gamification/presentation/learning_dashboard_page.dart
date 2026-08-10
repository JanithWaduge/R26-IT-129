import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../mastery/data/mastery_api.dart';
import '../../mastery/domain/mastery_models.dart';
import '../../review/data/review_api.dart';
import '../../review/domain/review_models.dart';
import '../data/gamification_api.dart';
import '../domain/gamification_models.dart';

class LearningDashboardPage extends StatefulWidget {
  const LearningDashboardPage({super.key});

  @override
  State<LearningDashboardPage> createState() => _LearningDashboardPageState();
}

class _LearningDashboardPageState extends State<LearningDashboardPage> {
  final GamificationApi _gamificationApi = GamificationApi();
  final MasteryApi _masteryApi = MasteryApi();
  final ReviewApi _reviewApi = ReviewApi();

  GamificationProfile? _profile;
  List<GamificationHistoryItem> _history = [];
  MasteryOverview? _mastery;
  ReviewOverview? _reviews;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _gamificationApi.dispose();
    _masteryApi.dispose();
    _reviewApi.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final results = await Future.wait<dynamic>([
        _gamificationApi.fetchProfile(),
        _gamificationApi.fetchHistory(),
        _masteryApi.fetchOverview(),
        _reviewApi.fetchOverview(),
      ]);
      if (!mounted) {
        return;
      }
      setState(() {
        _profile = results[0] as GamificationProfile;
        _history = results[1] as List<GamificationHistoryItem>;
        _mastery = results[2] as MasteryOverview;
        _reviews = results[3] as ReviewOverview;
        _loading = false;
      });
    } on ApiClientException catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = error.message;
        _loading = false;
      });
    } catch (_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = 'Unable to load the learning dashboard.';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Learning Dashboard')),
      body: SafeArea(
        child: RefreshIndicator(onRefresh: _load, child: _buildBody(context)),
      ),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_loading) {
      return ListView(
        children: const [
          SizedBox(height: 180),
          Center(child: CircularProgressIndicator()),
        ],
      );
    }
    if (_error != null) {
      return ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const SizedBox(height: 100),
          const Icon(Icons.error_outline, size: 72),
          const SizedBox(height: 16),
          Text(_error!, textAlign: TextAlign.center),
          const SizedBox(height: 16),
          FilledButton(onPressed: _load, child: const Text('Try Again')),
        ],
      );
    }

    final profile = _profile!;
    final mastery = _mastery!;
    final reviews = _reviews!;
    final unlockedBadges = profile.badges.where((badge) => badge.unlocked);

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Row(
                  children: [
                    CircleAvatar(
                      radius: 34,
                      child: Text(
                        profile.level.toString(),
                        style: Theme.of(context).textTheme.headlineSmall,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Level ${profile.level}',
                            style: Theme.of(context).textTheme.titleLarge,
                          ),
                          Text('${profile.totalXp} total XP'),
                        ],
                      ),
                    ),
                    Column(
                      children: [
                        const Icon(Icons.local_fire_department),
                        Text('${profile.currentStreakDays} days'),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                LinearProgressIndicator(value: profile.levelProgress),
                const SizedBox(height: 8),
                Text(
                  '${profile.xpIntoLevel} / ${profile.xpToNextLevel} '
                  'XP to next level',
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: _MetricCard(
                icon: Icons.quiz,
                label: 'Quizzes',
                value: profile.totalQuizzes.toString(),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _MetricCard(
                icon: Icons.check_circle,
                label: 'Accuracy',
                value: '${profile.accuracyPercentage.toStringAsFixed(1)}%',
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _MetricCard(
                icon: Icons.schedule,
                label: 'Reviews Due',
                value: reviews.receptiveDue.toString(),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _MetricCard(
                icon: Icons.psychology,
                label: 'Mastery',
                value: '${(mastery.averageReceptiveScore * 100).round()}%',
              ),
            ),
          ],
        ),
        const SizedBox(height: 24),
        Text('Achievements', style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 12),
        if (unlockedBadges.isEmpty)
          const Card(
            child: Padding(
              padding: EdgeInsets.all(20),
              child: Text(
                'Complete your first video quiz to unlock an achievement.',
                textAlign: TextAlign.center,
              ),
            ),
          )
        else
          ...unlockedBadges.map(
            (badge) => Card(
              child: ListTile(
                leading: const Icon(Icons.emoji_events),
                title: Text(badge.title),
                subtitle: Text(badge.description),
              ),
            ),
          ),
        const SizedBox(height: 24),
        Text(
          'Recent Quiz Rewards',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 12),
        if (_history.isEmpty)
          const Card(
            child: Padding(
              padding: EdgeInsets.all(20),
              child: Text(
                'No completed quizzes yet.',
                textAlign: TextAlign.center,
              ),
            ),
          )
        else
          ..._history.map(
            (item) => Card(
              child: ListTile(
                leading: const Icon(Icons.auto_awesome),
                title: Text('+${item.xpAwarded} XP'),
                subtitle: Text(
                  'Level ${item.levelAfter} • '
                  '${item.streakAfter}-day streak',
                ),
                trailing: Text('${item.createdAt.day}/${item.createdAt.month}'),
              ),
            ),
          ),
      ],
    );
  }
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Icon(icon),
            const SizedBox(height: 8),
            Text(value, style: Theme.of(context).textTheme.headlineSmall),
            Text(label, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}
