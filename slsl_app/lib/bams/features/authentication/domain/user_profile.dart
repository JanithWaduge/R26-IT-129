class UserProfile {
  const UserProfile({
    required this.userId,
    required this.studentId,
    required this.email,
    required this.role,
    required this.fullName,
    required this.preferredLanguage,
    required this.gradeLevel,
    required this.createdAt,
  });

  final String userId;
  final String studentId;
  final String email;
  final String role;
  final String fullName;
  final String preferredLanguage;
  final String gradeLevel;
  final DateTime createdAt;

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      userId: json['user_id'] as String,
      studentId: json['student_id'] as String,
      email: json['email'] as String,
      role: json['role'] as String,
      fullName: json['full_name'] as String,
      preferredLanguage: json['preferred_language'] as String,
      gradeLevel: json['grade_level'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
    );
  }
}
