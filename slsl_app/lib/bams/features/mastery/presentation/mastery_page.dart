import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../../core/theme/app_colors.dart';
import '../data/mastery_api.dart';
import '../domain/mastery_models.dart';

class MasteryPage extends StatefulWidget {
  const MasteryPage({required this.preferredLanguage, super.key});

  final String preferredLanguage;

  @override
  State<MasteryPage> createState() => _MasteryPageState();
}

class _MasteryPageState extends State<MasteryPage> {
  final MasteryApi _api = MasteryApi();

  MasteryOverview? _overview;
  PaginatedMastery? _mastery;

  bool _loading = true;
  String? _error;

  String? _statusFilter;

  @override
  void initState() {
    super.initState();

    _loadData();
  }

  @override
  void dispose() {
    _api.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final MasteryOverview overview = await _api.fetchOverview();

      final PaginatedMastery mastery = await _api.fetchSignMastery(
        status: _statusFilter,
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _overview = overview;
        _mastery = mastery;
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
        _error = 'Unable to load mastery data.';
        _loading = false;
      });
    }
  }

  String _formatStatus(String value) {
    return value
        .replaceAll('_', ' ')
        .split(' ')
        .map(
          (String word) =>
              word.isEmpty ? word : (word[0].toUpperCase() + word.substring(1)),
        )
        .join(' ');
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'very_weak':
        return AppColors.error;
      case 'weak':
        return AppColors.warning;
      case 'learning':
        return AppColors.primary;
      case 'proficient':
        return AppColors.secondary;
      case 'mastered':
        return AppColors.success;
      default: // 'new'
        return AppColors.inkSoft;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: AppTheme.light,
      child: Scaffold(
        backgroundColor: AppColors.background,
        appBar: AppBar(title: const Text('My Progress')),
        body: SafeArea(
          child: RefreshIndicator(
            onRefresh: _loadData,
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
          const SizedBox(height: 20),
          FilledButton(onPressed: _loadData, child: const Text('Try Again')),
        ],
      );
    }

    final MasteryOverview overview = _overview!;

    final List<SignMastery> items = _mastery?.items ?? [];

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        const Text(
          'Mastery Overview',
          style: TextStyle(color: AppColors.ink, fontSize: 20, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 16),
        _OverviewCard(
          title: 'Signs attempted',
          value:
              '${overview.attemptedSigns}'
              ' / ${overview.totalSigns}',
          progress: overview.totalSigns == 0
              ? 0
              : overview.attemptedSigns / overview.totalSigns,
          color: AppColors.primary,
        ),
        const SizedBox(height: 12),
        _OverviewCard(
          title: 'Receptive average',
          value: '${(overview.averageReceptiveScore * 100).round()}%',
          progress: overview.averageReceptiveScore,
          color: AppColors.success,
        ),
        const SizedBox(height: 12),
        _OverviewCard(
          title: 'Productive average',
          value: '${(overview.averageProductiveScore * 100).round()}%',
          progress: overview.averageProductiveScore,
          color: AppColors.secondary,
        ),
        const SizedBox(height: 12),
        _OverviewCard(
          title: 'Combined mastery',
          value: '${(overview.averageCombinedScore * 100).round()}%',
          progress: overview.averageCombinedScore,
          color: AppColors.indigo,
        ),
        const SizedBox(height: 20),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                _StatusChip(
                    label: 'New', count: overview.statuses.newCount, color: AppColors.inkSoft),
                _StatusChip(
                  label: 'Very weak',
                  count: overview.statuses.veryWeak,
                  color: AppColors.error,
                ),
                _StatusChip(
                    label: 'Weak', count: overview.statuses.weak, color: AppColors.warning),
                _StatusChip(
                  label: 'Learning',
                  count: overview.statuses.learning,
                  color: AppColors.primary,
                ),
                _StatusChip(
                  label: 'Proficient',
                  count: overview.statuses.proficient,
                  color: AppColors.secondary,
                ),
                _StatusChip(
                  label: 'Mastered',
                  count: overview.statuses.mastered,
                  color: AppColors.success,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 20),
        DropdownButtonFormField<String?>(
          value: _statusFilter,
          style: const TextStyle(color: AppColors.ink, fontSize: 15),
          dropdownColor: Colors.white,
          decoration: const InputDecoration(
            labelText: 'Mastery status filter',
          ),
          items: const [
            DropdownMenuItem<String?>(
              value: null,
              child: Text('All statuses', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem<String?>(
              value: 'new',
              child: Text('New', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem<String?>(
              value: 'very_weak',
              child: Text('Very weak', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem<String?>(
              value: 'weak',
              child: Text('Weak', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem<String?>(
              value: 'learning',
              child: Text('Learning', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem<String?>(
              value: 'proficient',
              child: Text('Proficient', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem<String?>(
              value: 'mastered',
              child: Text('Mastered', style: TextStyle(color: AppColors.ink)),
            ),
          ],
          onChanged: (String? value) {
            setState(() {
              _statusFilter = value;
            });

            _loadData();
          },
        ),
        const SizedBox(height: 24),
        const Text('Sign Mastery',
            style: TextStyle(color: AppColors.ink, fontSize: 18, fontWeight: FontWeight.w700)),
        const SizedBox(height: 12),
        if (items.isEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Text(
                'No signs match the selected status.',
                textAlign: TextAlign.center,
                style: const TextStyle(color: AppColors.inkSoft),
              ),
            ),
          )
        else
          ...items.map((SignMastery item) {
            final statusColor = _statusColor(item.overallStatus);
            return Card(
              margin: const EdgeInsets.only(bottom: 14),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            item.meanings.forLanguage(widget.preferredLanguage),
                            style: const TextStyle(
                                color: AppColors.ink, fontSize: 15, fontWeight: FontWeight.w700),
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                          decoration: BoxDecoration(
                            color: statusColor.withOpacity(0.14),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(_formatStatus(item.overallStatus),
                              style: TextStyle(
                                  color: statusColor,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700)),
                        ),
                      ],
                    ),
                    Text(item.gloss, style: const TextStyle(color: AppColors.inkSoft, fontSize: 12)),
                    const SizedBox(height: 16),
                    _MasteryBar(
                      label: 'Receptive',
                      score: item.receptive.score,
                      status: item.receptive.status,
                      color: AppColors.primary,
                    ),
                    const SizedBox(height: 12),
                    _MasteryBar(
                      label: 'Productive',
                      score: item.productive.score,
                      status: item.productive.status,
                      color: AppColors.secondary,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Balance: '
                      '${_formatStatus(item.balanceStatus)}',
                      style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
                    ),
                    Text(
                      'Reviews: '
                      '${item.receptive.totalReviews} receptive, '
                      '${item.productive.totalReviews} productive',
                      style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
                    ),
                    if (item.receptive.lastMlProbability != null) ...[
                      const SizedBox(height: 8),
                      Text(
                        'Predicted retention: '
                        '${(item.receptive.lastMlProbability! * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
                      ),
                      Text(
                        'SM-2 interval: ${item.receptive.baseSm2IntervalDays} days',
                        style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
                      ),
                      Text(
                        'Hybrid interval: ${item.receptive.intervalDays} days',
                        style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
                      ),
                    ],
                    if (item.receptive.lastMlModelVersion?.contains(
                          'synthetic',
                        ) ==
                        true)
                      const Text('Development prediction only',
                          style: TextStyle(color: AppColors.inkSoft, fontSize: 11)),
                  ],
                ),
              ),
            );
          }),
      ],
    );
  }
}

class _OverviewCard extends StatelessWidget {
  const _OverviewCard({
    required this.title,
    required this.value,
    required this.progress,
    required this.color,
  });

  final String title;
  final String value;
  final double progress;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                    child: Text(title, style: const TextStyle(color: AppColors.inkSoft))),
                Text(value,
                    style: TextStyle(color: color, fontSize: 16, fontWeight: FontWeight.w800)),
              ],
            ),
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: progress.clamp(0.0, 1.0),
                backgroundColor: AppColors.hairline,
                valueColor: AlwaysStoppedAnimation(color),
                minHeight: 7,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MasteryBar extends StatelessWidget {
  const _MasteryBar({
    required this.label,
    required this.score,
    required this.status,
    required this.color,
  });

  final String label;
  final double score;
  final String status;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
                child:
                    Text(label, style: const TextStyle(color: AppColors.ink, fontSize: 13))),
            Text(
              '${(score * 100).round()}% • '
              '${status.replaceAll('_', ' ')}',
              style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: score.clamp(0.0, 1.0),
            backgroundColor: AppColors.hairline,
            valueColor: AlwaysStoppedAnimation(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.label, required this.count, required this.color});

  final String label;
  final int count;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text('$label: $count',
          style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600)),
    );
  }
}