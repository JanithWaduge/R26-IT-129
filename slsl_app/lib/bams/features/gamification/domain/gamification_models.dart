class GamificationBadge {
  const GamificationBadge({
    required this.code,
    required this.title,
    required this.description,
    required this.iconKey,
    required this.unlocked,
    required this.unlockedAt,
  });

  final String code;
  final String title;
  final String description;
  final String iconKey;
  final bool unlocked;
  final DateTime? unlockedAt;

  factory GamificationBadge.fromJson(Map<String, dynamic> json) {
    return GamificationBadge(
      code: json['code'] as String,
      title: json['title'] as String,
      description: json['description'] as String,
      iconKey: json['icon_key'] as String,
      unlocked: json['unlocked'] as bool,
      unlockedAt: json['unlocked_at'] == null
          ? null
          : DateTime.parse(json['unlocked_at'] as String),
    );
  }
}

class GamificationRewardBreakdown {
  const GamificationRewardBreakdown({
    required this.questionXp,
    required this.dueReviewBonus,
    required this.completionBonus,
    required this.perfectBonus,
  });

  final int questionXp;
  final int dueReviewBonus;
  final int completionBonus;
  final int perfectBonus;

  factory GamificationRewardBreakdown.fromJson(Map<String, dynamic> json) {
    return GamificationRewardBreakdown(
      questionXp: json['question_xp'] as int,
      dueReviewBonus: json['due_review_bonus'] as int,
      completionBonus: json['completion_bonus'] as int,
      perfectBonus: json['perfect_bonus'] as int,
    );
  }
}

class GamificationReward {
  const GamificationReward({
    required this.xpAwarded,
    required this.totalXp,
    required this.level,
    required this.xpIntoLevel,
    required this.xpToNextLevel,
    required this.currentStreakDays,
    required this.badgesAwarded,
    required this.breakdown,
  });

  final int xpAwarded;
  final int totalXp;
  final int level;
  final int xpIntoLevel;
  final int xpToNextLevel;
  final int currentStreakDays;
  final List<GamificationBadge> badgesAwarded;
  final GamificationRewardBreakdown breakdown;

  factory GamificationReward.fromJson(Map<String, dynamic> json) {
    return GamificationReward(
      xpAwarded: json['xp_awarded'] as int,
      totalXp: json['total_xp'] as int,
      level: json['level'] as int,
      xpIntoLevel: json['xp_into_level'] as int,
      xpToNextLevel: json['xp_to_next_level'] as int,
      currentStreakDays: json['current_streak_days'] as int,
      badgesAwarded: (json['badges_awarded'] as List<dynamic>)
          .map(
            (dynamic item) =>
                GamificationBadge.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
      breakdown: GamificationRewardBreakdown.fromJson(
        json['breakdown'] as Map<String, dynamic>,
      ),
    );
  }
}

class GamificationProfile {
  const GamificationProfile({
    required this.totalXp,
    required this.level,
    required this.xpIntoLevel,
    required this.xpToNextLevel,
    required this.currentStreakDays,
    required this.longestStreakDays,
    required this.lastActivityDate,
    required this.totalQuizzes,
    required this.totalQuestions,
    required this.totalCorrect,
    required this.totalIncorrect,
    required this.totalSkipped,
    required this.totalHints,
    required this.totalRetries,
    required this.perfectQuizzes,
    required this.accuracyPercentage,
    required this.badges,
  });

  final int totalXp;
  final int level;
  final int xpIntoLevel;
  final int xpToNextLevel;
  final int currentStreakDays;
  final int longestStreakDays;
  final String? lastActivityDate;
  final int totalQuizzes;
  final int totalQuestions;
  final int totalCorrect;
  final int totalIncorrect;
  final int totalSkipped;
  final int totalHints;
  final int totalRetries;
  final int perfectQuizzes;
  final double accuracyPercentage;
  final List<GamificationBadge> badges;

  double get levelProgress {
    if (xpToNextLevel == 0) {
      return 0;
    }
    return (xpIntoLevel / xpToNextLevel).clamp(0.0, 1.0);
  }

  factory GamificationProfile.fromJson(Map<String, dynamic> json) {
    return GamificationProfile(
      totalXp: json['total_xp'] as int,
      level: json['level'] as int,
      xpIntoLevel: json['xp_into_level'] as int,
      xpToNextLevel: json['xp_to_next_level'] as int,
      currentStreakDays: json['current_streak_days'] as int,
      longestStreakDays: json['longest_streak_days'] as int,
      lastActivityDate: json['last_activity_date'] as String?,
      totalQuizzes: json['total_quizzes'] as int,
      totalQuestions: json['total_questions'] as int,
      totalCorrect: json['total_correct'] as int,
      totalIncorrect: json['total_incorrect'] as int,
      totalSkipped: json['total_skipped'] as int,
      totalHints: json['total_hints'] as int,
      totalRetries: json['total_retries'] as int,
      perfectQuizzes: json['perfect_quizzes'] as int,
      accuracyPercentage: (json['accuracy_percentage'] as num).toDouble(),
      badges: (json['badges'] as List<dynamic>)
          .map(
            (dynamic item) =>
                GamificationBadge.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
    );
  }
}

class GamificationHistoryItem {
  const GamificationHistoryItem({
    required this.sessionId,
    required this.xpAwarded,
    required this.badgesAwarded,
    required this.totalXpAfter,
    required this.levelAfter,
    required this.streakAfter,
    required this.createdAt,
  });

  final String sessionId;
  final int xpAwarded;
  final List<String> badgesAwarded;
  final int totalXpAfter;
  final int levelAfter;
  final int streakAfter;
  final DateTime createdAt;

  factory GamificationHistoryItem.fromJson(Map<String, dynamic> json) {
    return GamificationHistoryItem(
      sessionId: json['session_id'] as String,
      xpAwarded: json['xp_awarded'] as int,
      badgesAwarded: (json['badges_awarded'] as List<dynamic>).cast<String>(),
      totalXpAfter: json['total_xp_after'] as int,
      levelAfter: json['level_after'] as int,
      streakAfter: json['streak_after'] as int,
      createdAt: DateTime.parse(json['created_at'] as String),
    );
  }
}
