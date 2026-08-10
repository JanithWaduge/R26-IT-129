// lib/screens/adaptive_dashboard_screen_kisal.dart
//
// Renamed from Kisal's original views/dashboard_view.dart.
// Class renamed DashboardView -> AdaptiveDashboardScreenKisal to avoid any
// collision with other teammates' "dashboard" screens (e.g. Hansika's
// teacher_dashboard_screen). Imports updated to point at the shared
// project's services/screens locations.

import 'package:flutter/material.dart';
import '../services/adaptive_api_service_kisal.dart';
import 'quiz_screen_kisal.dart';

class AdaptiveDashboardScreenKisal extends StatefulWidget {
  const AdaptiveDashboardScreenKisal({super.key});

  @override
  State<AdaptiveDashboardScreenKisal> createState() =>
      _AdaptiveDashboardScreenKisalState();
}

class _AdaptiveDashboardScreenKisalState
    extends State<AdaptiveDashboardScreenKisal> {
  final AdaptiveApiServiceKisal _apiService = AdaptiveApiServiceKisal();
  bool _isLoading = true;
  Map<String, dynamic> _analyticsData = {};

  @override
  void initState() {
    super.initState();
    _loadDashboardData();
  }

  Future<void> _loadDashboardData() async {
    setState(() => _isLoading = true);
    try {
      final data = await _apiService.fetchDashboardAnalytics();
      if (!mounted) return;
      setState(() {
        _analyticsData = data;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Synchronization error: $e')));
    }
  }

  double _asDouble(dynamic value) {
    if (value is num) return value.toDouble();
    return double.tryParse(value?.toString() ?? '') ?? 0.0;
  }

  int _asInt(dynamic value) {
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '') ?? 0;
  }

  List<dynamic> _asList(dynamic value) {
    return value is List ? value : <dynamic>[];
  }

  Map<String, dynamic> _asStringMap(dynamic value) {
    if (value is Map<String, dynamic>) return value;
    if (value is Map) {
      return value.map((key, mapValue) => MapEntry(key.toString(), mapValue));
    }
    return <String, dynamic>{};
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final mastery = _asStringMap(
      _analyticsData['category_mastery_percentages'],
    );
    final sm2Summary = _asStringMap(_analyticsData['sm2_summary']);
    final collectedData = _asStringMap(_analyticsData['collected_data']);
    final reviewQueue = _asList(_analyticsData['review_queue']);

    return Scaffold(
      backgroundColor: const Color(0xFFF6F7FB),
      appBar: AppBar(
        title: const Text(
          'SLSL Learning Dashboard',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: const Color(0xFFF6F7FB),
        actions: [
          IconButton(
            tooltip: 'Refresh dashboard',
            icon: const Icon(Icons.refresh),
            onPressed: _loadDashboardData,
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadDashboardData,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildHeroSummary(sm2Summary),
              const SizedBox(height: 14),
              _buildStatGrid(sm2Summary, collectedData),
              const SizedBox(height: 18),
              _buildSectionTitle('Curriculum Progress'),
              _buildProgressPanel(mastery),
              const SizedBox(height: 18),
              _buildSectionTitle('SM-2 Adaptive Process'),
              _buildSm2Flow(sm2Summary),
              const SizedBox(height: 18),
              _buildSectionTitle('Collected Learning Data'),
              _buildCollectedDataPanel(collectedData),
              const SizedBox(height: 18),
              _buildSectionTitle('Review Schedule'),
              _buildReviewQueue(reviewQueue),
              const SizedBox(height: 22),
              _buildQuizActions(sm2Summary),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeroSummary(Map<String, dynamic> sm2Summary) {
    final dueNow = _asInt(sm2Summary['due_now_count']);
    final scheduled = _asInt(sm2Summary['scheduled_count']);

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFF132238),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Container(
            width: 54,
            height: 54,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(
              Icons.psychology_alt,
              color: Colors.white,
              size: 30,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Adaptive memory engine',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '$dueNow signs due now, $scheduled scheduled for later',
                  style: const TextStyle(
                    color: Color(0xFFD7E0EA),
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatGrid(
    Map<String, dynamic> sm2Summary,
    Map<String, dynamic> collectedData,
  ) {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      mainAxisSpacing: 10,
      crossAxisSpacing: 10,
      childAspectRatio: 1.55,
      children: [
        _buildMetricCard(
          icon: Icons.local_fire_department,
          label: 'Study streak',
          value: '${_analyticsData['active_login_streak'] ?? 0} days',
          color: Colors.deepOrange,
        ),
        _buildMetricCard(
          icon: Icons.verified,
          label: 'Mastered',
          value: '${_analyticsData['total_signs_mastered'] ?? 0} signs',
          color: Colors.teal,
        ),
        _buildMetricCard(
          icon: Icons.speed,
          label: 'Avg ease factor',
          value: _asDouble(
            sm2Summary['average_ease_factor'],
          ).toStringAsFixed(2),
          color: Colors.indigo,
        ),
        _buildMetricCard(
          icon: Icons.storage,
          label: 'Tracked records',
          value: '${collectedData['tracked_sign_records'] ?? 0}',
          color: Colors.blueGrey,
        ),
      ],
    );
  }

  Widget _buildMetricCard({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Icon(icon, color: color, size: 24),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  value,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  label,
                  style: TextStyle(color: Colors.grey[650], fontSize: 12),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Text(
        title,
        style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
      ),
    );
  }

  Widget _buildProgressPanel(Map<String, dynamic> mastery) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _buildCategoryProgressRow(
              'School Items',
              _asDouble(mastery['school']),
              Colors.blue,
            ),
            _buildCategoryProgressRow(
              'Numbers Module',
              _asDouble(mastery['numbers']),
              Colors.green,
            ),
            _buildCategoryProgressRow(
              'Daily Routine',
              _asDouble(mastery['daily_life']),
              Colors.purple,
            ),
            _buildCategoryProgressRow(
              'Emotions',
              _asDouble(mastery['emotions']),
              Colors.pink,
            ),
            _buildCategoryProgressRow(
              'Greetings',
              _asDouble(mastery['greetings']),
              Colors.orange,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSm2Flow(Map<String, dynamic> sm2Summary) {
    final steps = _asList(sm2Summary['algorithm_steps']);
    final qualityScale = _asList(sm2Summary['quality_scale']);

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: _buildSm2MiniMetric(
                    'Avg interval',
                    '${_asDouble(sm2Summary['average_interval_days']).toStringAsFixed(1)} days',
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _buildSm2MiniMetric(
                    'Due now',
                    '${sm2Summary['due_now_count'] ?? 0}',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            for (int i = 0; i < steps.length; i++)
              _buildFlowStep(i + 1, steps[i].toString(), i == steps.length - 1),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: qualityScale.map((item) {
                final data = _asStringMap(item);
                return Chip(
                  avatar: CircleAvatar(
                    backgroundColor: const Color(0xFF132238),
                    child: Text(
                      '${data['score'] ?? '-'}',
                      style: const TextStyle(color: Colors.white, fontSize: 11),
                    ),
                  ),
                  label: Text(data['meaning']?.toString() ?? ''),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSm2MiniMetric(String label, String value) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFEAF0F7),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: TextStyle(color: Colors.grey[700], fontSize: 12)),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
          ),
        ],
      ),
    );
  }

  Widget _buildFlowStep(int number, String text, bool isLast) {
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Column(
            children: [
              CircleAvatar(
                radius: 14,
                backgroundColor: const Color(0xFF132238),
                child: Text(
                  '$number',
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                ),
              ),
              if (!isLast)
                Expanded(
                  child: Container(width: 2, color: const Color(0xFFDCE3EC)),
                ),
            ],
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: 14),
              child: Text(text, style: const TextStyle(height: 1.35)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCollectedDataPanel(Map<String, dynamic> collectedData) {
    final fields = _asList(collectedData['stored_tracker_fields']);

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildDataLine(
              'Student ID',
              collectedData['student_id']?.toString() ?? '-',
            ),
            _buildDataLine(
              'Encountered signs',
              '${_analyticsData['total_signs_encountered'] ?? 0}',
            ),
            _buildDataLine(
              'Stored tracker records',
              '${collectedData['tracked_sign_records'] ?? 0}',
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: fields
                  .map((field) => Chip(label: Text(field.toString())))
                  .toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDataLine(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Expanded(
            child: Text(label, style: TextStyle(color: Colors.grey[700])),
          ),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildReviewQueue(List<dynamic> reviewQueue) {
    if (reviewQueue.isEmpty) {
      return const Card(
        elevation: 0,
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
            'No collected review records yet. Complete a quiz to create SM-2 data.',
          ),
        ),
      );
    }

    return Column(
      children: reviewQueue.take(6).map((item) {
        final data = _asStringMap(item);
        final isDue = data['is_due'] == true;
        return Card(
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          child: ListTile(
            leading: CircleAvatar(
              backgroundColor: isDue ? Colors.green[100] : Colors.blueGrey[100],
              child: Icon(
                isDue ? Icons.notifications_active : Icons.event_available,
                color: isDue ? Colors.green[800] : Colors.blueGrey[800],
              ),
            ),
            title: Text(
              data['word_english']?.toString() ??
                  data['sign_id']?.toString() ??
                  'Sign',
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
            subtitle: Text(
              'EF ${data['ease_factor'] ?? 0}  |  reps ${data['repetitions'] ?? 0}  |  interval ${data['interval_days'] ?? 0}d',
            ),
            trailing: Text(
              isDue ? 'Due' : 'Later',
              style: TextStyle(
                color: isDue ? Colors.green[800] : Colors.blueGrey[700],
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildQuizActions(Map<String, dynamic> sm2Summary) {
    final dueNow = _asInt(sm2Summary['due_now_count']);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ElevatedButton.icon(
          icon: const Icon(Icons.psychology_alt, size: 24),
          label: Text(
            dueNow > 0
                ? 'Start Adaptive Review ($dueNow due)'
                : 'No Adaptive Reviews Due',
            style: const TextStyle(fontSize: 16),
          ),
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
            backgroundColor: const Color(0xFF1769E0),
            foregroundColor: Colors.white,
            disabledBackgroundColor: Colors.blueGrey[100],
            disabledForegroundColor: Colors.blueGrey[500],
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          onPressed: dueNow == 0
              ? null
              : () => _openQuiz(includePractice: false),
        ),
        const SizedBox(height: 10),
        OutlinedButton.icon(
          icon: const Icon(Icons.replay_circle_filled),
          label: const Text('Practice School Signs Anyway'),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          onPressed: () => _openQuiz(includePractice: true),
        ),
      ],
    );
  }

  Future<void> _openQuiz({required bool includePractice}) async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => QuizViewKisal(
          category: 'school',
          includePractice: includePractice,
        ),
      ),
    );
    _loadDashboardData();
  }

  Widget _buildCategoryProgressRow(
    String title,
    double percentage,
    Color displayColor,
  ) {
    final safePercentage = percentage.clamp(0.0, 100.0);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 9),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
              Text('${safePercentage.toStringAsFixed(1)}%'),
            ],
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: safePercentage / 100.0,
            minHeight: 10,
            backgroundColor: Colors.grey[200],
            valueColor: AlwaysStoppedAnimation<Color>(displayColor),
            borderRadius: BorderRadius.circular(5),
          ),
        ],
      ),
    );
  }
}