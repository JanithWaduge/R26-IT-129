// lib/models/sign_item_kisal.dart
//
// Renamed from Kisal's original models/sign_item.dart (class renamed
// SignItem -> SignItemKisal to avoid any future naming collisions with
// other teammates' models in the shared project).

class SignItemKisal {
  final String signId;
  final String wordEnglish;
  final String wordSinhala;
  final String wordTamil;
  final String category;
  final String avatarAssetPath;

  SignItemKisal({
    required this.signId,
    required this.wordEnglish,
    required this.wordSinhala,
    required this.wordTamil,
    required this.category,
    required this.avatarAssetPath,
  });

  // Factory to convert incoming MongoDB JSON properties directly into memory
  factory SignItemKisal.fromJson(Map<String, dynamic> json) {
    return SignItemKisal(
      signId: json['sign_id'] as String,
      wordEnglish: json['word_english'] as String,
      wordSinhala: json['word_sinhala'] as String,
      wordTamil: json['word_tamil'] as String,
      category: json['category'] as String,
      avatarAssetPath: json['avatar_asset_path'] as String,
    );
  }
}