class SignMedia {
  const SignMedia({
    required this.sourceType,
    required this.uri,
    required this.thumbnailUri,
    required this.durationMs,
    required this.objectiveSource,
  });

  final String sourceType;
  final String? uri;
  final String? thumbnailUri;
  final int? durationMs;
  final String objectiveSource;

  bool get isAvailable =>
      sourceType != 'pending' && uri != null && uri!.isNotEmpty;

  factory SignMedia.fromJson(Map<String, dynamic> json) {
    return SignMedia(
      sourceType: json['source_type'] as String,
      uri: json['uri'] as String?,
      thumbnailUri: json['thumbnail_uri'] as String?,
      durationMs: json['duration_ms'] as int?,
      objectiveSource: json['objective_source'] as String,
    );
  }
}
