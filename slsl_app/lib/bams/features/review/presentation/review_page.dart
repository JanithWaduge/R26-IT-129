import 'package:flutter/material.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../../../core/theme/app_colors.dart';
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
    return Theme(
      data: AppTheme.light,
      child: Scaffold(
        backgroundColor: AppColors.background,
        appBar: AppBar(title: const Text('Review Due')),
        body: SafeArea(
          child: RefreshIndicator(
              onRefresh: _load, color: AppColors.primary, child: _body(context)),
        ),
      ),
    );
  }

  Widget _body(BuildContext context) {
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
          Text(_error!,
              textAlign: TextAlign.center, style: const TextStyle(color: AppColors.inkSoft)),
          const SizedBox(height: 12),
          FilledButton(onPressed: _load, child: const Text('Try Again')),
        ],
      );
    }
    final overview = _overview!;
    final due = _dueReviews!;
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(22),
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
              Text(
                '${overview.totalDueDirections}',
                style: const TextStyle(
                    color: Colors.white, fontSize: 40, fontWeight: FontWeight.w800),
              ),
              const Text('Review directions due',
                  style: TextStyle(color: Colors.white70, fontSize: 13)),
              const SizedBox(height: 10),
              Text(
                'Receptive ${overview.receptiveDue} • Productive ${overview.productiveDue} • Overdue ${overview.overdueDirections}',
                style: const TextStyle(color: Colors.white, fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        DropdownButtonFormField<String>(
          value: _mode,
          style: const TextStyle(color: AppColors.ink, fontSize: 15),
          dropdownColor: Colors.white,
          decoration: const InputDecoration(
            labelText: 'Review mode',
            prefixIcon: Icon(Icons.tune, color: AppColors.secondary),
          ),
          items: const [
            DropdownMenuItem(
              value: 'mixed',
              child: Text('Mixed', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem(
              value: 'receptive',
              child: Text('Receptive', style: TextStyle(color: AppColors.ink)),
            ),
            DropdownMenuItem(
              value: 'productive',
              child: Text('Productive', style: TextStyle(color: AppColors.ink)),
            ),
          ],
          onChanged: (value) {
            if (value != null) {
              _mode = value;
              _load();
            }
          },
        ),
        const SizedBox(height: 16),
        SizedBox(
          height: 52,
          child: DecoratedBox(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              gradient: (due.totalItems == 0 || _starting) ? null : AppColors.primaryGradient,
              color: (due.totalItems == 0 || _starting) ? AppColors.hairline : null,
            ),
            child: FilledButton.icon(
              onPressed: due.totalItems == 0 || _starting ? null : _startReview,
              style: FilledButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                disabledBackgroundColor: Colors.transparent,
              ),
              icon: const Icon(Icons.replay),
              label: Text(
                _starting
                    ? 'Starting...'
                    : 'Start Due Review (${due.totalItems > 5 ? 5 : due.totalItems})',
              ),
            ),
          ),
        ),
        const SizedBox(height: 20),
        ...due.items.map(
          (item) => Card(
            margin: const EdgeInsets.only(bottom: 10),
            child: ListTile(
              leading: Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: AppColors.warning.withOpacity(0.14),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.schedule, color: AppColors.warning, size: 20),
              ),
              title: Text(item.meanings.forLanguage(widget.preferredLanguage),
                  style: const TextStyle(color: AppColors.ink, fontWeight: FontWeight.w600)),
              subtitle: Text(
                '${item.direction} • ${_label(item.selectionReason)}\nMastery ${(item.masteryScore * 100).round()}% • Priority ${item.priorityScore.toStringAsFixed(1)}',
                style: const TextStyle(color: AppColors.inkSoft, fontSize: 12),
              ),
            ),
          ),
        ),
      ],
    );
  }
}