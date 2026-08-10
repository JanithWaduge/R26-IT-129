import '../../../core/network/authenticated_api_client.dart';
import '../domain/curriculum_category.dart';
import '../domain/curriculum_competency.dart';
import '../domain/paginated_signs.dart';
import '../domain/sign_detail.dart';

class VocabularyApi {
  VocabularyApi({AuthenticatedApiClient? apiClient})
    : _apiClient = apiClient ?? AuthenticatedApiClient();

  final AuthenticatedApiClient _apiClient;

  Future<List<CurriculumCategory>> fetchCategories() async {
    final dynamic decoded = await _apiClient.getJson('/curriculum/categories');

    if (decoded is! List<dynamic>) {
      throw const ApiClientException('Invalid category response.');
    }

    return decoded
        .map(
          (dynamic item) =>
              CurriculumCategory.fromJson(item as Map<String, dynamic>),
        )
        .toList();
  }

  Future<List<CurriculumCompetency>> fetchCompetencies({
    String? categoryId,
  }) async {
    final dynamic decoded = await _apiClient.getJson(
      '/curriculum/competencies',
      queryParameters: categoryId == null ? null : {'category_id': categoryId},
    );

    if (decoded is! List<dynamic>) {
      throw const ApiClientException('Invalid competency response.');
    }

    return decoded
        .map(
          (dynamic item) =>
              CurriculumCompetency.fromJson(item as Map<String, dynamic>),
        )
        .toList();
  }

  Future<PaginatedSigns> fetchSigns({
    int page = 1,
    int pageSize = 20,
    String? categoryId,
    String? competencyId,
    int? difficulty,
    String? search,
  }) async {
    final Map<String, String> query = {
      'page': page.toString(),
      'page_size': pageSize.toString(),
    };

    if (categoryId != null) {
      query['category_id'] = categoryId;
    }

    if (competencyId != null) {
      query['competency_id'] = competencyId;
    }

    if (difficulty != null) {
      query['difficulty'] = difficulty.toString();
    }

    if (search != null && search.trim().isNotEmpty) {
      query['search'] = search.trim();
    }

    final dynamic decoded = await _apiClient.getJson(
      '/signs',
      queryParameters: query,
    );

    if (decoded is! Map<String, dynamic>) {
      throw const ApiClientException('Invalid vocabulary response.');
    }

    return PaginatedSigns.fromJson(decoded);
  }

  Future<SignDetail> fetchSign(String signId) async {
    final dynamic decoded = await _apiClient.getJson('/signs/$signId');

    if (decoded is! Map<String, dynamic>) {
      throw const ApiClientException('Invalid sign response.');
    }

    return SignDetail.fromJson(decoded);
  }

  void dispose() {
    _apiClient.dispose();
  }
}
