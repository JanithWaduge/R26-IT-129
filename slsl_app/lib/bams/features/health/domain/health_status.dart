class HealthStatus {
  const HealthStatus({
    required this.status,
    required this.service,
    required this.version,
    required this.environment,
    required this.database,
    required this.timestamp,
  });

  final String status;
  final String service;
  final String version;
  final String environment;
  final String database;
  final DateTime timestamp;

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    return HealthStatus(
      status: json['status'] as String? ?? 'unknown',
      service: json['service'] as String? ?? 'Unknown service',
      version: json['version'] as String? ?? 'Unknown version',
      environment: (json['environment'] as String? ?? 'unknown'),
      database: (json['database'] as String? ?? 'unknown'),
      timestamp: DateTime.parse(json['timestamp'] as String),
    );
  }
}
