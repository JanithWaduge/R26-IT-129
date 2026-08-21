import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import '../data/health_service.dart';
import '../domain/health_status.dart';

class HealthPage extends StatefulWidget {
  const HealthPage({super.key});

  @override
  State<HealthPage> createState() => _HealthPageState();
}

class _HealthPageState extends State<HealthPage> {
  final HealthService _healthService = HealthService();

  late Future<HealthStatus> _healthFuture;

  @override
  void initState() {
    super.initState();
    _healthFuture = _healthService.fetchHealth();
  }

  @override
  void dispose() {
    _healthService.dispose();
    super.dispose();
  }

  void _retryConnection() {
    setState(() {
      _healthFuture = _healthService.fetchHealth();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('SLSL-BAMS')),
      body: SafeArea(
        child: FutureBuilder<HealthStatus>(
          future: _healthFuture,
          builder:
              (BuildContext context, AsyncSnapshot<HealthStatus> snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(color: AppColors.primary),
                        SizedBox(height: 16),
                        Text('Connecting to the research backend...',
                            style: TextStyle(color: AppColors.inkSoft)),
                      ],
                    ),
                  );
                }

                if (snapshot.hasError) {
                  return Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Container(
                            width: 96,
                            height: 96,
                            decoration: BoxDecoration(
                              color: AppColors.error.withOpacity(0.12),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.cloud_off,
                                size: 48, color: AppColors.error),
                          ),
                          const SizedBox(height: 16),
                          const Text(
                            'Backend connection failed',
                            style: TextStyle(
                                color: AppColors.ink,
                                fontSize: 18,
                                fontWeight: FontWeight.w700),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            snapshot.error.toString(),
                            style: const TextStyle(color: AppColors.inkSoft),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 24),
                          FilledButton.icon(
                            onPressed: _retryConnection,
                            style: FilledButton.styleFrom(backgroundColor: AppColors.error),
                            icon: const Icon(Icons.refresh),
                            label: const Text('Try Again'),
                          ),
                        ],
                      ),
                    ),
                  );
                }

                final HealthStatus health = snapshot.data!;

                return RefreshIndicator(
                  onRefresh: () async {
                    _retryConnection();
                    await _healthFuture;
                  },
                  color: AppColors.primary,
                  child: ListView(
                    padding: const EdgeInsets.all(24),
                    children: [
                      const SizedBox(height: 40),
                      Container(
                        width: 104,
                        height: 104,
                        margin: const EdgeInsets.symmetric(horizontal: 0),
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          color: AppColors.success.withOpacity(0.12),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.check_circle,
                            size: 56, color: AppColors.success),
                      ),
                      const SizedBox(height: 20),
                      const Text(
                        'Backend Connected',
                        style: TextStyle(
                            color: AppColors.ink, fontSize: 22, fontWeight: FontWeight.w800),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        'The Python backend and Flutter Android app '
                        'are communicating successfully.',
                        style: TextStyle(color: AppColors.inkSoft),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 32),
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(20),
                          child: Column(
                            children: [
                              _InformationRow(
                                label: 'Status',
                                value: health.status,
                              ),
                              const Divider(),
                              _InformationRow(
                                label: 'Service',
                                value: health.service,
                              ),
                              const Divider(),
                              _InformationRow(
                                label: 'Version',
                                value: health.version,
                              ),
                              const Divider(),
                              _InformationRow(
                                label: 'Environment',
                                value: health.environment,
                              ),
                              const Divider(),
                              _InformationRow(
                                label: 'Database',
                                value: health.database,
                              ),
                              const Divider(),
                              _InformationRow(
                                label: 'Server Time',
                                value: health.timestamp.toLocal().toString(),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                      FilledButton.icon(
                        onPressed: _retryConnection,
                        icon: const Icon(Icons.sync),
                        label: const Text('Check Connection Again'),
                      ),
                    ],
                  ),
                );
              },
        ),
      ),
    );
  }
}

class _InformationRow extends StatelessWidget {
  const _InformationRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.bold, color: AppColors.ink),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
              flex: 2,
              child: Text(value,
                  textAlign: TextAlign.end,
                  style: const TextStyle(color: AppColors.inkSoft))),
        ],
      ),
    );
  }
}