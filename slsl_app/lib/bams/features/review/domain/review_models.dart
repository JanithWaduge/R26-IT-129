import '../../vocabulary/domain/localized_text.dart';

class DueReviewItem {
  const DueReviewItem({
    required this.signId,
    required this.code,
    required this.gloss,
    required this.meanings,
    required this.direction,
    required this.difficulty,
    required this.masteryScore,
    required this.masteryStatus,
    required this.easeFactor,
    required this.intervalDays,
    required this.repetitionCount,
    required this.lapseCount,
    required this.nextReviewAt,
    required this.isDue,
    required this.daysOverdue,
    required this.priorityScore,
    required this.selectionReason,
    required this.curriculumWeight,
  });

  final String signId;
  final String code;
  final String gloss;
  final LocalizedText meanings;
  final String direction;
  final int difficulty;
  final double masteryScore;
  final String masteryStatus;
  final double easeFactor;
  final int intervalDays;
  final int repetitionCount;
  final int lapseCount;
  final DateTime? nextReviewAt;
  final bool isDue;
  final double daysOverdue;
  final double priorityScore;
  final String selectionReason;
  final double curriculumWeight;

  factory DueReviewItem.fromJson(Map<String, dynamic> json) => DueReviewItem(
    signId: json['sign_id'] as String,
    code: json['code'] as String,
    gloss: json['gloss'] as String,
    meanings: LocalizedText.fromJson(json['meanings'] as Map<String, dynamic>),
    direction: json['direction'] as String,
    difficulty: json['difficulty'] as int,
    masteryScore: (json['mastery_score'] as num).toDouble(),
    masteryStatus: json['mastery_status'] as String,
    easeFactor: (json['ease_factor'] as num).toDouble(),
    intervalDays: json['interval_days'] as int,
    repetitionCount: json['repetition_count'] as int,
    lapseCount: json['lapse_count'] as int,
    nextReviewAt: json['next_review_at'] == null
        ? null
        : DateTime.parse(json['next_review_at'] as String),
    isDue: json['is_due'] as bool,
    daysOverdue: (json['days_overdue'] as num).toDouble(),
    priorityScore: (json['priority_score'] as num).toDouble(),
    selectionReason: json['selection_reason'] as String,
    curriculumWeight: (json['curriculum_weight'] as num).toDouble(),
  );
}

class ReviewOverview {
  const ReviewOverview({
    required this.totalSigns,
    required this.receptiveDue,
    required this.productiveDue,
    required this.totalDueDirections,
    required this.newReceptiveDirections,
    required this.newProductiveDirections,
    required this.overdueDirections,
    required this.nextScheduledReviewAt,
    required this.highestPriorityScore,
  });

  final int totalSigns;
  final int receptiveDue;
  final int productiveDue;
  final int totalDueDirections;
  final int newReceptiveDirections;
  final int newProductiveDirections;
  final int overdueDirections;
  final DateTime? nextScheduledReviewAt;
  final double highestPriorityScore;

  factory ReviewOverview.fromJson(Map<String, dynamic> json) => ReviewOverview(
    totalSigns: json['total_signs'] as int,
    receptiveDue: json['receptive_due'] as int,
    productiveDue: json['productive_due'] as int,
    totalDueDirections: json['total_due_directions'] as int,
    newReceptiveDirections: json['new_receptive_directions'] as int,
    newProductiveDirections: json['new_productive_directions'] as int,
    overdueDirections: json['overdue_directions'] as int,
    nextScheduledReviewAt: json['next_scheduled_review_at'] == null
        ? null
        : DateTime.parse(json['next_scheduled_review_at'] as String),
    highestPriorityScore: (json['highest_priority_score'] as num).toDouble(),
  );
}

class DueReviewList {
  const DueReviewList({
    required this.items,
    required this.totalItems,
    required this.mode,
  });
  final List<DueReviewItem> items;
  final int totalItems;
  final String mode;

  factory DueReviewList.fromJson(Map<String, dynamic> json) => DueReviewList(
    items: (json['items'] as List<dynamic>)
        .map((item) => DueReviewItem.fromJson(item as Map<String, dynamic>))
        .toList(),
    totalItems: json['total_items'] as int,
    mode: json['mode'] as String,
  );
}
