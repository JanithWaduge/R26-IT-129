import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../../core/theme/app_colors.dart';
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
    return Theme(
      data: AppTheme.light,
      child: Scaffold(
        backgroundColor: AppColors.background,
        appBar: AppBar(title: const Text('Learning Dashboard')),
        body: SafeArea(
          child: RefreshIndicator(
            onRefresh: _load,
            color: AppColors.primary,
            child: _buildBody(context),
          ),
        ),
      ),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_loading) {
      return ListView(
        children: const [
          SizedBox(height: 180),
          Center(child: CircularProgressIndicator(color: AppColors.primary)),
        ],
      );
    }
    if (_error != null) {
      return ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const SizedBox(height: 100),
          const Icon(Icons.error_outline, size: 72, color: AppColors.error),
          const SizedBox(height: 16),
          Text(_error!,
              textAlign: TextAlign.center, style: const TextStyle(color: AppColors.inkSoft)),
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
        // ── Level card: gradient hero, matching the profile header ──
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
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
              Row(
                children: [
                  Container(
                    width: 64,
                    height: 64,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.18),
                      shape: BoxShape.circle,
                    ),
                    child: Center(
                      child: Text(
                        profile.level.toString(),
                        style: const TextStyle(
                            color: Colors.white, fontSize: 24, fontWeight: FontWeight.w800),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Level ${profile.level}',
                          style: const TextStyle(
                              color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700),
                        ),
                        Text('${profile.totalXp} total XP',
                            style: const TextStyle(color: Colors.white70, fontSize: 12)),
                      ],
                    ),
                  ),
                  Column(
                    children: [
                      const Icon(Icons.local_fire_department, color: Colors.white),
                      Text('${profile.currentStreakDays} days',
                          style: const TextStyle(color: Colors.white, fontSize: 12)),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 20),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(
                  value: profile.levelProgress,
                  backgroundColor: Colors.white24,
                  valueColor: const AlwaysStoppedAnimation(Colors.white),
                  minHeight: 6,
                ),
              ),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '${profile.xpIntoLevel} / ${profile.xpToNextLevel} '
                  'XP to next level',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ),
            ],
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
                color: AppColors.primary,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _MetricCard(
                icon: Icons.check_circle,
                label: 'Accuracy',
                value: '${profile.accuracyPercentage.toStringAsFixed(1)}%',
                color: AppColors.success,
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
                color: AppColors.warning,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _MetricCard(
                icon: Icons.psychology,
                label: 'Mastery',
                value: '${(mastery.averageReceptiveScore * 100).round()}%',
                color: AppColors.secondary,
              ),
            ),
          ],
        ),
        const SizedBox(height: 24),
        const Text('Achievements',
            style: TextStyle(color: AppColors.ink, fontSize: 18, fontWeight: FontWeight.w700)),
        const SizedBox(height: 12),
        if (unlockedBadges.isEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Text(
                'Complete your first video quiz to unlock an achievement.',
                textAlign: TextAlign.center,
                style: const TextStyle(color: AppColors.inkSoft),
              ),
            ),
          )
        else
          ...unlockedBadges.map(
            (badge) => Card(
              margin: const EdgeInsets.only(bottom: 10),
              child: ListTile(
                leading: Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: AppColors.warning.withOpacity(0.14),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.emoji_events, color: AppColors.warning),
                ),
                title: Text(badge.title,
                    style: const TextStyle(color: AppColors.ink, fontWeight: FontWeight.w700)),
                subtitle: Text(badge.description,
                    style: const TextStyle(color: AppColors.inkSoft)),
              ),
            ),
          ),
        const SizedBox(height: 24),
        const Text(
          'Recent Quiz Rewards',
          style: TextStyle(color: AppColors.ink, fontSize: 18, fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 12),
        if (_history.isEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Text(
                'No completed quizzes yet.',
                textAlign: TextAlign.center,
                style: const TextStyle(color: AppColors.inkSoft),
              ),
            ),
          )
        else
          ..._history.map(
            (item) => Card(
              margin: const EdgeInsets.only(bottom: 10),
              child: ListTile(
                leading: Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: AppColors.success.withOpacity(0.14),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.auto_awesome, color: AppColors.success),
                ),
                title: Text('+${item.xpAwarded} XP',
                    style: const TextStyle(color: AppColors.ink, fontWeight: FontWeight.w700)),
                subtitle: Text(
                  'Level ${item.levelAfter} • '
                  '${item.streakAfter}-day streak',
                  style: const TextStyle(color: AppColors.inkSoft),
                ),
                trailing: Text('${item.createdAt.day}/${item.createdAt.month}',
                    style: const TextStyle(color: AppColors.inkSoft, fontSize: 12)),
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
    required this.color,
  });

  final IconData icon;
  final String label;
  final String value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(color: color.withOpacity(0.14), shape: BoxShape.circle),
              child: Icon(icon, color: color, size: 18),
            ),
            const SizedBox(height: 10),
            Text(value,
                style:
                    TextStyle(color: AppColors.ink, fontSize: 20, fontWeight: FontWeight.w800)),
            Text(label,
                textAlign: TextAlign.center,
                style: const TextStyle(color: AppColors.inkSoft, fontSize: 12)),
          ],
        ),
      ),
    );
  }
}