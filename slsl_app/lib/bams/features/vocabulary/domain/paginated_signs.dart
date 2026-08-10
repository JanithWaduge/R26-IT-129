import 'sign_summary.dart';

class PaginatedSigns {
  const PaginatedSigns({
    required this.items,
    required this.page,
    required this.pageSize,
    required this.totalItems,
    required this.totalPages,
  });

  final List<SignSummary> items;
  final int page;
  final int pageSize;
  final int totalItems;
  final int totalPages;

  factory PaginatedSigns.fromJson(Map<String, dynamic> json) {
    return PaginatedSigns(
      items: (json['items'] as List<dynamic>)
          .map(
            (dynamic item) =>
                SignSummary.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
      page: json['page'] as int,
      pageSize: json['page_size'] as int,
      totalItems: json['total_items'] as int,
      totalPages: json['total_pages'] as int,
    );
  }
}
