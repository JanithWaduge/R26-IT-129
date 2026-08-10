import '../../../core/network/authenticated_api_client.dart';
import '../domain/gamification_models.dart';

class GamificationApi {
  GamificationApi({AuthenticatedApiClient? apiClient})
    : _apiClient = apiClient ?? AuthenticatedApiClient();

  final AuthenticatedApiClient _apiClient;

  Future<GamificationProfile> fetchProfile() async {
    final dynamic decoded = await _apiClient.getJson('/gamification/profile');
    return GamificationProfile.fromJson(decoded as Map<String, dynamic>);
  }

  Future<List<GamificationHistoryItem>> fetchHistory({int limit = 10}) async {
    final dynamic decoded = await _apiClient.getJson(
      '/gamification/history',
      queryParameters: {'limit': limit.toString()},
    );
    final Map<String, dynamic> body = decoded as Map<String, dynamic>;
    return (body['items'] as List<dynamic>)
        .map(
          (dynamic item) =>
              GamificationHistoryItem.fromJson(item as Map<String, dynamic>),
        )
        .toList();
  }

  void dispose() {
    _apiClient.dispose();
  }
}
