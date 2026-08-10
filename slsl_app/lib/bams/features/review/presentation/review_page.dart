import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../quiz/presentation/quiz_page.dart';
import '../data/review_api.dart';
import '../domain/review_models.dart';

class ReviewPage extends StatefulWidget {
  const ReviewPage({required this.preferredLanguage, super.key});
  final String preferredLanguage;

  @override
  State<ReviewPage> createState() => _ReviewPageState();
}

class _ReviewPageState extends State<ReviewPage> {
  final ReviewApi _api = ReviewApi();
  ReviewOverview? _overview;
  DueReviewList? _dueReviews;
  String _mode = 'mixed';
  bool _loading = true;
  bool _starting = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _api.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final overview = await _api.fetchOverview();
      final due = await _api.fetchDue(mode: _mode);
      if (!mounted) {
        return;
      }
      setState(() {
        _overview = overview;
        _dueReviews = due;
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
    }
  }

  Future<void> _startReview() async {
    final available = _dueReviews?.totalItems ?? 0;
    if (available == 0) {
      return;
    }
    setState(() => _starting = true);
    try {
      final session = await _api.startDueReview(
        promptLanguage: widget.preferredLanguage,
        mode: _mode,
        questionCount: available > 5 ? 5 : available,
      );
      if (!mounted) {
        return;
      }
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => QuizPage(initialSession: session),
        ),
      );
      if (mounted) {
        await _load();
      }
    } on ApiClientException catch (error) {
      if (mounted) {
        setState(() => _error = error.message);
      }
    } finally {
      if (mounted) {
        setState(() => _starting = false);
      }
    }
  }

  String _label(String value) => value
      .replaceAll('_', ' ')
      .split(' ')
      .map(
        (word) => word.isEmpty
            ? word
            : '${word[0].toUpperCase()}${word.substring(1)}',
      )
      .join(' ');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Review Due')),
      body: SafeArea(
        child: RefreshIndicator(onRefresh: _load, child: _body(context)),
      ),
    );
  }

  Widget _body(BuildContext context) {
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
          Text(_error!, textAlign: TextAlign.center),
          FilledButton(onPressed: _load, child: const Text('Try Again')),
        ],
      );
    }
    final overview = _overview!;
    final due = _dueReviews!;
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Text(
                  '${overview.totalDueDirections}',
                  style: Theme.of(context).textTheme.displayMedium,
                ),
                const Text('Review directions due'),
                Text(
                  'Receptive ${overview.receptiveDue} • Productive ${overview.productiveDue} • Overdue ${overview.overdueDirections}',
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        DropdownButtonFormField<String>(
          value: _mode,
          decoration: const InputDecoration(
            labelText: 'Review mode',
            border: OutlineInputBorder(),
          ),
          items: const [
            DropdownMenuItem(value: 'mixed', child: Text('Mixed')),
            DropdownMenuItem(value: 'receptive', child: Text('Receptive')),
            DropdownMenuItem(value: 'productive', child: Text('Productive')),
          ],
          onChanged: (value) {
            if (value != null) {
              _mode = value;
              _load();
            }
          },
        ),
        const SizedBox(height: 16),
        FilledButton.icon(
          onPressed: due.totalItems == 0 || _starting ? null : _startReview,
          icon: const Icon(Icons.replay),
          label: Text(
            _starting
                ? 'Starting...'
                : 'Start Due Review (${due.totalItems > 5 ? 5 : due.totalItems})',
          ),
        ),
        const SizedBox(height: 20),
        ...due.items.map(
          (item) => Card(
            child: ListTile(
              title: Text(item.meanings.forLanguage(widget.preferredLanguage)),
              subtitle: Text(
                '${item.direction} • ${_label(item.selectionReason)}\nMastery ${(item.masteryScore * 100).round()}% • Priority ${item.priorityScore.toStringAsFixed(1)}',
              ),
            ),
          ),
        ),
      ],
    );
  }
}