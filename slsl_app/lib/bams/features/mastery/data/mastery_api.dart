import '../../../core/network/authenticated_api_client.dart';
import '../domain/mastery_models.dart';

class MasteryApi {
  MasteryApi({AuthenticatedApiClient? apiClient})
    : _apiClient = apiClient ?? AuthenticatedApiClient();

  final AuthenticatedApiClient _apiClient;

  Future<MasteryOverview> fetchOverview() async {
    final dynamic decoded = await _apiClient.getJson('/mastery/overview');

    return MasteryOverview.fromJson(decoded as Map<String, dynamic>);
  }

  Future<PaginatedMastery> fetchSignMastery({
    String? status,
    String? categoryId,
  }) async {
    final Map<String, String> query = {'page': '1', 'page_size': '100'};

    if (status != null) {
      query['mastery_status'] = status;
    }

    if (categoryId != null) {
      query['category_id'] = categoryId;
    }

    final dynamic decoded = await _apiClient.getJson(
      '/mastery/signs',
      queryParameters: query,
    );

    return PaginatedMastery.fromJson(decoded as Map<String, dynamic>);
  }

  Future<SignMastery> fetchOneSign(String signId) async {
    final dynamic decoded = await _apiClient.getJson('/mastery/signs/$signId');

    return SignMastery.fromJson(decoded as Map<String, dynamic>);
  }

  void dispose() {
    _apiClient.dispose();
  }
}
