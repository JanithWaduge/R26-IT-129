import 'curriculum_category.dart';
import 'curriculum_competency.dart';
import 'localized_text.dart';
import 'sign_media.dart';
import 'sign_summary.dart';

class SignDetail {
  const SignDetail({
    required this.id,
    required this.code,
    required this.gloss,
    required this.meanings,
    required this.category,
    required this.competencies,
    required this.difficulty,
    required this.tags,
    required this.prerequisites,
    required this.media,
    required this.contentStatus,
    required this.validationStatus,
  });

  final String id;
  final String code;
  final String gloss;
  final LocalizedText meanings;
  final CurriculumCategory category;
  final List<CurriculumCompetency> competencies;
  final int difficulty;
  final List<String> tags;
  final List<SignSummary> prerequisites;
  final SignMedia media;
  final String contentStatus;
  final String validationStatus;

  factory SignDetail.fromJson(Map<String, dynamic> json) {
    return SignDetail(
      id: json['id'] as String,
      code: json['code'] as String,
      gloss: json['gloss'] as String,
      meanings: LocalizedText.fromJson(
        json['meanings'] as Map<String, dynamic>,
      ),
      category: CurriculumCategory.fromJson(
        json['category'] as Map<String, dynamic>,
      ),
      competencies: (json['competencies'] as List<dynamic>)
          .map(
            (dynamic item) =>
                CurriculumCompetency.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
      difficulty: json['difficulty'] as int,
      tags: (json['tags'] as List<dynamic>).cast<String>(),
      prerequisites: (json['prerequisites'] as List<dynamic>)
          .map(
            (dynamic item) =>
                SignSummary.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
      media: SignMedia.fromJson(json['media'] as Map<String, dynamic>),
      contentStatus: json['content_status'] as String,
      validationStatus: json['validation_status'] as String,
    );
  }
}
