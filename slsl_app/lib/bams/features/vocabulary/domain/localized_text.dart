class LocalizedText {
  const LocalizedText({
    required this.english,
    required this.sinhala,
    required this.tamil,
  });

  final String english;
  final String sinhala;
  final String tamil;

  factory LocalizedText.fromJson(Map<String, dynamic> json) {
    return LocalizedText(
      english: json['english'] as String,
      sinhala: json['sinhala'] as String,
      tamil: json['tamil'] as String,
    );
  }

  String forLanguage(String language) {
    switch (language) {
      case 'sinhala':
        return sinhala;
      case 'tamil':
        return tamil;
      default:
        return english;
    }
  }
}
