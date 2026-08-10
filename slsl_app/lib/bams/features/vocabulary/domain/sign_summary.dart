import 'localized_text.dart';
import 'sign_media.dart';

class SignSummary {
  const SignSummary({
    required this.id,
    required this.code,
    required this.gloss,
    required this.meanings,
    required this.categoryId,
    required this.difficulty,
    required this.media,
    required this.contentStatus,
    required this.validationStatus,
  });

  final String id;
  final String code;
  final String gloss;
  final LocalizedText meanings;
  final String categoryId;
  final int difficulty;
  final SignMedia media;
  final String contentStatus;
  final String validationStatus;

  factory SignSummary.fromJson(Map<String, dynamic> json) {
    return SignSummary(
      id: json['id'] as String,
      code: json['code'] as String,
      gloss: json['gloss'] as String,
      meanings: LocalizedText.fromJson(
        json['meanings'] as Map<String, dynamic>,
      ),
      categoryId: json['category_id'] as String,
      difficulty: json['difficulty'] as int,
      media: SignMedia.fromJson(json['media'] as Map<String, dynamic>),
      contentStatus: json['content_status'] as String,
      validationStatus: json['validation_status'] as String,
    );
  }
}
