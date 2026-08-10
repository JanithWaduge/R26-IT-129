import '../../../core/network/authenticated_api_client.dart';
import '../../quiz/data/quiz_api.dart';
import '../../quiz/domain/quiz_session.dart';
import '../domain/review_models.dart';

class ReviewApi {
  ReviewApi({AuthenticatedApiClient? apiClient, QuizApi? quizApi})
    : _apiClient = apiClient ?? AuthenticatedApiClient(),
      _quizApi = quizApi ?? QuizApi();

  final AuthenticatedApiClient _apiClient;
  final QuizApi _quizApi;

  Future<ReviewOverview> fetchOverview() async {
    final dynamic decoded = await _apiClient.getJson('/reviews/overview');
    return ReviewOverview.fromJson(decoded as Map<String, dynamic>);
  }

  Future<DueReviewList> fetchDue({
    String mode = 'mixed',
    int limit = 20,
  }) async {
    final dynamic decoded = await _apiClient.getJson(
      '/reviews/due',
      queryParameters: {'mode': mode, 'limit': limit.toString()},
    );
    return DueReviewList.fromJson(decoded as Map<String, dynamic>);
  }

  Future<QuizSession> startDueReview({
    required String promptLanguage,
    required String mode,
    required int questionCount,
  }) => _quizApi.createSession(
    promptLanguage: promptLanguage,
    questionCount: questionCount,
    mode: mode,
    selectionStrategy: 'due_only',
  );

  void dispose() {
    _apiClient.dispose();
    _quizApi.dispose();
  }
}
