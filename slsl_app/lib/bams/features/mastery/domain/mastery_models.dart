import '../../vocabulary/domain/localized_text.dart';

class DirectionMastery {
  const DirectionMastery({
    required this.score,
    required this.status,
    required this.totalReviews,
    required this.successfulReviews,
    required this.independentSuccesses,
    required this.assistedSuccesses,
    required this.failureCount,
    required this.averageQualityScore,
    required this.lastQualityScore,
    required this.averageResponseTimeMs,
    required this.averageRecognitionConfidence,
    required this.baseSm2IntervalDays,
    required this.intervalDays,
    required this.lastMlProbability,
    required this.lastMlAdjustmentFactor,
    required this.lastMlModelVersion,
    required this.lastMlStatus,
  });

  final double score;
  final String status;

  final int totalReviews;
  final int successfulReviews;
  final int independentSuccesses;
  final int assistedSuccesses;
  final int failureCount;

  final double averageQualityScore;
  final double? lastQualityScore;
  final double? averageResponseTimeMs;
  final double? averageRecognitionConfidence;
  final int baseSm2IntervalDays;
  final int intervalDays;
  final double? lastMlProbability;
  final double? lastMlAdjustmentFactor;
  final String? lastMlModelVersion;
  final String lastMlStatus;

  factory DirectionMastery.fromJson(Map<String, dynamic> json) {
    return DirectionMastery(
      score: (json['score'] as num).toDouble(),
      status: json['status'] as String,
      totalReviews: json['total_reviews'] as int,
      successfulReviews: json['successful_reviews'] as int,
      independentSuccesses: json['independent_successes'] as int,
      assistedSuccesses: json['assisted_successes'] as int,
      failureCount: json['failure_count'] as int,
      averageQualityScore: (json['average_quality_score'] as num).toDouble(),
      lastQualityScore: json['last_quality_score'] == null
          ? null
          : (json['last_quality_score'] as num).toDouble(),
      averageResponseTimeMs: json['average_response_time_ms'] == null
          ? null
          : (json['average_response_time_ms'] as num).toDouble(),
      averageRecognitionConfidence:
          json['average_recognition_confidence'] == null
          ? null
          : (json['average_recognition_confidence'] as num).toDouble(),
      baseSm2IntervalDays: json['base_sm2_interval_days'] as int,
      intervalDays: json['interval_days'] as int,
      lastMlProbability: json['last_ml_probability'] == null
          ? null
          : (json['last_ml_probability'] as num).toDouble(),
      lastMlAdjustmentFactor: json['last_ml_adjustment_factor'] == null
          ? null
          : (json['last_ml_adjustment_factor'] as num).toDouble(),
      lastMlModelVersion: json['last_ml_model_version'] as String?,
      lastMlStatus: json['last_ml_status'] as String,
    );
  }
}

class SignMastery {
  const SignMastery({
    required this.signId,
    required this.code,
    required this.gloss,
    required this.meanings,
    required this.categoryId,
    required this.difficulty,
    required this.receptive,
    required this.productive,
    required this.combinedScore,
    required this.overallStatus,
    required this.balanceStatus,
    required this.attempted,
  });

  final String signId;
  final String code;
  final String gloss;
  final LocalizedText meanings;
  final String categoryId;
  final int difficulty;

  final DirectionMastery receptive;
  final DirectionMastery productive;

  final double combinedScore;
  final String overallStatus;
  final String balanceStatus;
  final bool attempted;

  factory SignMastery.fromJson(Map<String, dynamic> json) {
    return SignMastery(
      signId: json['sign_id'] as String,
      code: json['code'] as String,
      gloss: json['gloss'] as String,
      meanings: LocalizedText.fromJson(
        json['meanings'] as Map<String, dynamic>,
      ),
      categoryId: json['category_id'] as String,
      difficulty: json['difficulty'] as int,
      receptive: DirectionMastery.fromJson(
        json['receptive'] as Map<String, dynamic>,
      ),
      productive: DirectionMastery.fromJson(
        json['productive'] as Map<String, dynamic>,
      ),
      combinedScore: (json['combined_score'] as num).toDouble(),
      overallStatus: json['overall_status'] as String,
      balanceStatus: json['balance_status'] as String,
      attempted: json['attempted'] as bool,
    );
  }
}

class MasteryStatusCounts {
  const MasteryStatusCounts({
    required this.newCount,
    required this.veryWeak,
    required this.weak,
    required this.learning,
    required this.proficient,
    required this.mastered,
  });

  final int newCount;
  final int veryWeak;
  final int weak;
  final int learning;
  final int proficient;
  final int mastered;

  factory MasteryStatusCounts.fromJson(Map<String, dynamic> json) {
    return MasteryStatusCounts(
      newCount: json['new'] as int,
      veryWeak: json['very_weak'] as int,
      weak: json['weak'] as int,
      learning: json['learning'] as int,
      proficient: json['proficient'] as int,
      mastered: json['mastered'] as int,
    );
  }
}

class MasteryOverview {
  const MasteryOverview({
    required this.totalSigns,
    required this.attemptedSigns,
    required this.unattemptedSigns,
    required this.receptiveAttempted,
    required this.productiveAttempted,
    required this.bidirectionallyAttempted,
    required this.averageReceptiveScore,
    required this.averageProductiveScore,
    required this.averageCombinedScore,
    required this.statuses,
    required this.receptiveWeakerCount,
    required this.productiveWeakerCount,
    required this.balancedCount,
  });

  final int totalSigns;
  final int attemptedSigns;
  final int unattemptedSigns;

  final int receptiveAttempted;
  final int productiveAttempted;
  final int bidirectionallyAttempted;

  final double averageReceptiveScore;
  final double averageProductiveScore;
  final double averageCombinedScore;

  final MasteryStatusCounts statuses;

  final int receptiveWeakerCount;
  final int productiveWeakerCount;
  final int balancedCount;

  factory MasteryOverview.fromJson(Map<String, dynamic> json) {
    return MasteryOverview(
      totalSigns: json['total_signs'] as int,
      attemptedSigns: json['attempted_signs'] as int,
      unattemptedSigns: json['unattempted_signs'] as int,
      receptiveAttempted: json['receptive_attempted'] as int,
      productiveAttempted: json['productive_attempted'] as int,
      bidirectionallyAttempted: json['bidirectionally_attempted'] as int,
      averageReceptiveScore: (json['average_receptive_score'] as num)
          .toDouble(),
      averageProductiveScore: (json['average_productive_score'] as num)
          .toDouble(),
      averageCombinedScore: (json['average_combined_score'] as num).toDouble(),
      statuses: MasteryStatusCounts.fromJson(
        json['statuses'] as Map<String, dynamic>,
      ),
      receptiveWeakerCount: json['receptive_weaker_count'] as int,
      productiveWeakerCount: json['productive_weaker_count'] as int,
      balancedCount: json['balanced_count'] as int,
    );
  }
}

class PaginatedMastery {
  const PaginatedMastery({
    required this.items,
    required this.page,
    required this.pageSize,
    required this.totalItems,
    required this.totalPages,
  });

  final List<SignMastery> items;
  final int page;
  final int pageSize;
  final int totalItems;
  final int totalPages;

  factory PaginatedMastery.fromJson(Map<String, dynamic> json) {
    return PaginatedMastery(
      items: (json['items'] as List<dynamic>)
          .map(
            (dynamic item) =>
                SignMastery.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
      page: json['page'] as int,
      pageSize: json['page_size'] as int,
      totalItems: json['total_items'] as int,
      totalPages: json['total_pages'] as int,
    );
  }
}
