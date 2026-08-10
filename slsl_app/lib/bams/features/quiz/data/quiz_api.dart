import 'package:uuid/uuid.dart';

import '../../../core/network/authenticated_api_client.dart';
import '../domain/quiz_session.dart';

class QuizApi {
  QuizApi({AuthenticatedApiClient? apiClient})
    : _apiClient = apiClient ?? AuthenticatedApiClient();

  final AuthenticatedApiClient _apiClient;

  static const Uuid _uuid = Uuid();

  Future<QuizSession> createSession({
    required String promptLanguage,
    required int questionCount,
    String mode = 'receptive',
    String selectionStrategy = 'adaptive',
    String? categoryId,
    int? difficulty,
  }) async {
    final dynamic decoded = await _apiClient.postJson(
      '/quizzes/sessions',
      body: {
        'client_request_id': _uuid.v4(),
        'mode': 'receptive',
        'selection_strategy': selectionStrategy,
        'prompt_language': promptLanguage,
        'question_count': questionCount,
        'category_id': categoryId,
        'competency_id': null,
        'difficulty': difficulty,
      },
    );

    return QuizSession.fromJson(decoded as Map<String, dynamic>);
  }

  Future<QuizActionResult> submitReceptiveAnswer({
    required String sessionId,
    required String questionId,
    required String selectedSignId,
    required int responseTimeMs,
  }) async {
    final dynamic decoded = await _apiClient.postJson(
      '/quizzes/sessions/'
      '$sessionId/answer',
      body: {
        'submission_id': _uuid.v4(),
        'question_id': questionId,
        'direction': 'receptive',
        'selected_sign_id': selectedSignId,
        'recognition_result': null,
        'client_response_time_ms': responseTimeMs,
      },
    );

    return QuizActionResult.fromJson(decoded as Map<String, dynamic>);
  }

  Future<QuizActionResult> skipQuestion({
    required String sessionId,
    required String questionId,
  }) async {
    final dynamic decoded = await _apiClient.postJson(
      '/quizzes/sessions/'
      '$sessionId/skip',
      body: {'action_id': _uuid.v4(), 'question_id': questionId},
    );

    return QuizActionResult.fromJson(decoded as Map<String, dynamic>);
  }

  Future<String> requestHint({
    required String sessionId,
    required String questionId,
  }) async {
    final dynamic decoded = await _apiClient.postJson(
      '/quizzes/sessions/'
      '$sessionId/hint',
      body: {'action_id': _uuid.v4(), 'question_id': questionId},
    );

    final Map<String, dynamic> body = decoded as Map<String, dynamic>;

    return body['message'] as String;
  }

  Future<QuizSession> abandon({required String sessionId}) async {
    final dynamic decoded = await _apiClient.postJson(
      '/quizzes/sessions/'
      '$sessionId/abandon',
    );

    return QuizSession.fromJson(decoded as Map<String, dynamic>);
  }

  void dispose() {
    _apiClient.dispose();
  }
}
