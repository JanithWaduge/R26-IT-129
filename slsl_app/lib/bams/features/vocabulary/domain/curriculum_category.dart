import 'localized_text.dart';

class CurriculumCategory {
  const CurriculumCategory({
    required this.id,
    required this.code,
    required this.name,
    required this.description,
    required this.displayOrder,
    required this.iconKey,
    required this.validationStatus,
  });

  final String id;
  final String code;
  final LocalizedText name;
  final LocalizedText description;
  final int displayOrder;
  final String iconKey;
  final String validationStatus;

  factory CurriculumCategory.fromJson(Map<String, dynamic> json) {
    return CurriculumCategory(
      id: json['id'] as String,
      code: json['code'] as String,
      name: LocalizedText.fromJson(json['name'] as Map<String, dynamic>),
      description: LocalizedText.fromJson(
        json['description'] as Map<String, dynamic>,
      ),
      displayOrder: json['display_order'] as int,
      iconKey: json['icon_key'] as String,
      validationStatus: json['validation_status'] as String,
    );
  }
}
