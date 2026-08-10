import '../../gamification/domain/gamification_models.dart';

class QuizOption {
  const QuizOption({required this.signId, required this.text});

  final String signId;
  final String text;

  factory QuizOption.fromJson(Map<String, dynamic> json) {
    return QuizOption(
      signId: json['sign_id'] as String,
      text: json['text'] as String,
    );
  }
}

class QuizQuestion {
  const QuizQuestion({
    required this.questionId,
    required this.direction,
    required this.orderIndex,
    required this.promptText,
    required this.mediaSourceType,
    required this.mediaUri,
    required this.categoryName,
    required this.difficulty,
    required this.options,
    required this.attemptNumber,
    required this.remainingAttempts,
    required this.hintUsed,
    required this.startedAt,
    required this.selectionPriority,
    required this.selectionReason,
    required this.wasDue,
    required this.daysOverdue,
  });

  final String questionId;
  final String direction;
  final int orderIndex;

  final String? promptText;
  final String? mediaSourceType;
  final String? mediaUri;

  final String categoryName;
  final int difficulty;

  final List<QuizOption> options;

  final int attemptNumber;
  final int remainingAttempts;
  final bool hintUsed;

  final DateTime startedAt;
  final double selectionPriority;
  final String selectionReason;
  final bool wasDue;
  final double daysOverdue;

  factory QuizQuestion.fromJson(Map<String, dynamic> json) {
    final dynamic media = json['media'];

    return QuizQuestion(
      questionId: json['question_id'] as String,
      direction: json['direction'] as String,
      orderIndex: json['order_index'] as int,
      promptText: json['prompt_text'] as String?,
      mediaSourceType: media is Map<String, dynamic>
          ? media['source_type'] as String?
          : null,
      mediaUri: media is Map<String, dynamic> ? media['uri'] as String? : null,
      categoryName: json['category_name'] as String,
      difficulty: json['difficulty'] as int,
      options: (json['options'] as List<dynamic>)
          .map(
            (dynamic item) => QuizOption.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
      attemptNumber: json['attempt_number'] as int,
      remainingAttempts: json['remaining_attempts'] as int,
      hintUsed: json['hint_used'] as bool,
      startedAt: DateTime.parse(json['started_at'] as String),
      selectionPriority: (json['selection_priority'] as num).toDouble(),
      selectionReason: json['selection_reason'] as String,
      wasDue: json['was_due'] as bool,
      daysOverdue: (json['days_overdue'] as num).toDouble(),
    );
  }
}

class QuizSummary {
  const QuizSummary({
    required this.correctCount,
    required this.incorrectCount,
    required this.skippedCount,
    required this.hintsUsed,
    required this.totalRetries,
    required this.receptiveCorrect,
    required this.productiveCorrect,
  });

  final int correctCount;
  final int incorrectCount;
  final int skippedCount;
  final int hintsUsed;
  final int totalRetries;
  final int receptiveCorrect;
  final int productiveCorrect;

  factory QuizSummary.fromJson(Map<String, dynamic> json) {
    return QuizSummary(
      correctCount: json['correct_count'] as int,
      incorrectCount: json['incorrect_count'] as int,
      skippedCount: json['skipped_count'] as int,
      hintsUsed: json['hints_used'] as int,
      totalRetries: json['total_retries'] as int,
      receptiveCorrect: json['receptive_correct'] as int,
      productiveCorrect: json['productive_correct'] as int,
    );
  }
}

class QuizSession {
  const QuizSession({
    required this.id,
    required this.mode,
    required this.selectionStrategy,
    required this.promptLanguage,
    required this.status,
    required this.questionCount,
    required this.currentQuestionIndex,
    required this.answeredCount,
    required this.progressPercentage,
    required this.summary,
    required this.currentQuestion,
    required this.startedAt,
    required this.expiresAt,
    required this.completedAt,
    required this.version,
  });

  final String id;
  final String mode;
  final String selectionStrategy;
  final String promptLanguage;
  final String status;

  final int questionCount;
  final int currentQuestionIndex;
  final int answeredCount;
  final double progressPercentage;

  final QuizSummary summary;
  final QuizQuestion? currentQuestion;

  final DateTime startedAt;
  final DateTime expiresAt;
  final DateTime? completedAt;

  final int version;

  bool get isFinished => status != 'in_progress';

  factory QuizSession.fromJson(Map<String, dynamic> json) {
    return QuizSession(
      id: json['id'] as String,
      mode: json['mode'] as String,
      selectionStrategy: json['selection_strategy'] as String,
      promptLanguage: json['prompt_language'] as String,
      status: json['status'] as String,
      questionCount: json['question_count'] as int,
      currentQuestionIndex: json['current_question_index'] as int,
      answeredCount: json['answered_count'] as int,
      progressPercentage: (json['progress_percentage'] as num).toDouble(),
      summary: QuizSummary.fromJson(json['summary'] as Map<String, dynamic>),
      currentQuestion: json['current_question'] == null
          ? null
          : QuizQuestion.fromJson(
              json['current_question'] as Map<String, dynamic>,
            ),
      startedAt: DateTime.parse(json['started_at'] as String),
      expiresAt: DateTime.parse(json['expires_at'] as String),
      completedAt: json['completed_at'] == null
          ? null
          : DateTime.parse(json['completed_at'] as String),
      version: json['version'] as int,
    );
  }
}

class QuizFeedback {
  const QuizFeedback({
    required this.action,
    required this.isCorrect,
    required this.questionFinalized,
    required this.retryAllowed,
    required this.message,
    required this.recognitionConfidence,
  });

  final String action;
  final bool? isCorrect;
  final bool questionFinalized;
  final bool retryAllowed;
  final String message;
  final double? recognitionConfidence;

  factory QuizFeedback.fromJson(Map<String, dynamic> json) {
    return QuizFeedback(
      action: json['action'] as String,
      isCorrect: json['is_correct'] as bool?,
      questionFinalized: json['question_finalized'] as bool,
      retryAllowed: json['retry_allowed'] as bool,
      message: json['message'] as String,
      recognitionConfidence: json['recognition_confidence'] == null
          ? null
          : (json['recognition_confidence'] as num).toDouble(),
    );
  }
}

class QuizActionResult {
  const QuizActionResult({
    required this.session,
    required this.feedback,
    required this.reward,
  });

  final QuizSession session;
  final QuizFeedback feedback;
  final GamificationReward? reward;

  factory QuizActionResult.fromJson(Map<String, dynamic> json) {
    return QuizActionResult(
      session: QuizSession.fromJson(json['session'] as Map<String, dynamic>),
      feedback: QuizFeedback.fromJson(json['feedback'] as Map<String, dynamic>),
      reward: json['reward'] == null
          ? null
          : GamificationReward.fromJson(json['reward'] as Map<String, dynamic>),
    );
  }
}
