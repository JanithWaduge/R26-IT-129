import 'localized_text.dart';

class CurriculumCompetency {
  const CurriculumCompetency({
    required this.id,
    required this.categoryId,
    required this.code,
    required this.title,
    required this.description,
    required this.gradeLevels,
    required this.receptiveThreshold,
    required this.productiveThreshold,
    required this.validationStatus,
  });

  final String id;
  final String categoryId;
  final String code;
  final LocalizedText title;
  final LocalizedText description;
  final List<String> gradeLevels;
  final double receptiveThreshold;
  final double productiveThreshold;
  final String validationStatus;

  factory CurriculumCompetency.fromJson(Map<String, dynamic> json) {
    return CurriculumCompetency(
      id: json['id'] as String,
      categoryId: json['category_id'] as String,
      code: json['code'] as String,
      title: LocalizedText.fromJson(json['title'] as Map<String, dynamic>),
      description: LocalizedText.fromJson(
        json['description'] as Map<String, dynamic>,
      ),
      gradeLevels: (json['grade_levels'] as List<dynamic>).cast<String>(),
      receptiveThreshold: (json['receptive_mastery_threshold'] as num)
          .toDouble(),
      productiveThreshold: (json['productive_mastery_threshold'] as num)
          .toDouble(),
      validationStatus: json['validation_status'] as String,
    );
  }
}
