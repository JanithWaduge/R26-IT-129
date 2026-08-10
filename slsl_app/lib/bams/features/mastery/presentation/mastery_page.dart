import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My Progress')),
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _loadData,
          child: _buildBody(context),
        ),
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
        Text(
          'Mastery Overview',
          style: Theme.of(context).textTheme.headlineSmall,
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
        ),
        const SizedBox(height: 12),
        _OverviewCard(
          title: 'Receptive average',
          value: '${(overview.averageReceptiveScore * 100).round()}%',
          progress: overview.averageReceptiveScore,
        ),
        const SizedBox(height: 12),
        _OverviewCard(
          title: 'Productive average',
          value: '${(overview.averageProductiveScore * 100).round()}%',
          progress: overview.averageProductiveScore,
        ),
        const SizedBox(height: 12),
        _OverviewCard(
          title: 'Combined mastery',
          value: '${(overview.averageCombinedScore * 100).round()}%',
          progress: overview.averageCombinedScore,
        ),
        const SizedBox(height: 20),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                _StatusChip(label: 'New', count: overview.statuses.newCount),
                _StatusChip(
                  label: 'Very weak',
                  count: overview.statuses.veryWeak,
                ),
                _StatusChip(label: 'Weak', count: overview.statuses.weak),
                _StatusChip(
                  label: 'Learning',
                  count: overview.statuses.learning,
                ),
                _StatusChip(
                  label: 'Proficient',
                  count: overview.statuses.proficient,
                ),
                _StatusChip(
                  label: 'Mastered',
                  count: overview.statuses.mastered,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 20),
        DropdownButtonFormField<String?>(
          value: _statusFilter,
          decoration: const InputDecoration(
            labelText: 'Mastery status filter',
            border: OutlineInputBorder(),
          ),
          items: const [
            DropdownMenuItem<String?>(value: null, child: Text('All statuses')),
            DropdownMenuItem<String?>(value: 'new', child: Text('New')),
            DropdownMenuItem<String?>(
              value: 'very_weak',
              child: Text('Very weak'),
            ),
            DropdownMenuItem<String?>(value: 'weak', child: Text('Weak')),
            DropdownMenuItem<String?>(
              value: 'learning',
              child: Text('Learning'),
            ),
            DropdownMenuItem<String?>(
              value: 'proficient',
              child: Text('Proficient'),
            ),
            DropdownMenuItem<String?>(
              value: 'mastered',
              child: Text('Mastered'),
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
        Text('Sign Mastery', style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 12),
        if (items.isEmpty)
          const Card(
            child: Padding(
              padding: EdgeInsets.all(24),
              child: Text(
                'No signs match the selected status.',
                textAlign: TextAlign.center,
              ),
            ),
          )
        else
          ...items.map((SignMastery item) {
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
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                        ),
                        Chip(label: Text(_formatStatus(item.overallStatus))),
                      ],
                    ),
                    Text(item.gloss),
                    const SizedBox(height: 16),
                    _MasteryBar(
                      label: 'Receptive',
                      score: item.receptive.score,
                      status: item.receptive.status,
                    ),
                    const SizedBox(height: 12),
                    _MasteryBar(
                      label: 'Productive',
                      score: item.productive.score,
                      status: item.productive.status,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Balance: '
                      '${_formatStatus(item.balanceStatus)}',
                    ),
                    Text(
                      'Reviews: '
                      '${item.receptive.totalReviews} receptive, '
                      '${item.productive.totalReviews} productive',
                    ),
                    if (item.receptive.lastMlProbability != null) ...[
                      const SizedBox(height: 8),
                      Text(
                        'Predicted retention: '
                        '${(item.receptive.lastMlProbability! * 100).toStringAsFixed(1)}%',
                      ),
                      Text(
                        'SM-2 interval: ${item.receptive.baseSm2IntervalDays} days',
                      ),
                      Text(
                        'Hybrid interval: ${item.receptive.intervalDays} days',
                      ),
                    ],
                    if (item.receptive.lastMlModelVersion?.contains(
                          'synthetic',
                        ) ==
                        true)
                      const Text('Development prediction only'),
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
  });

  final String title;
  final String value;
  final double progress;

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
                Expanded(child: Text(title)),
                Text(value, style: Theme.of(context).textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 10),
            LinearProgressIndicator(value: progress.clamp(0.0, 1.0)),
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
  });

  final String label;
  final double score;
  final String status;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(child: Text(label)),
            Text(
              '${(score * 100).round()}% • '
              '${status.replaceAll('_', ' ')}',
            ),
          ],
        ),
        const SizedBox(height: 6),
        LinearProgressIndicator(value: score.clamp(0.0, 1.0)),
      ],
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.label, required this.count});

  final String label;
  final int count;

  @override
  Widget build(BuildContext context) {
    return Chip(label: Text('$label: $count'));
  }
}
