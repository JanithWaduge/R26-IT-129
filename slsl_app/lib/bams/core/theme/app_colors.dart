import 'package:flutter/material.dart';

/// Shared brand palette - white background, blue / violet / green accents.
/// Mirrors the palette already applied across the SLSL Recognition app,
/// so both modules read as one product.
class AppColors {
  AppColors._();

  static const Color background = Color(0xFFFFFFFF);
  static const Color surface = Color(0xFFF6F8FC);
  static const Color hairline = Color(0xFFEDEFF5);

  static const Color ink = Color(0xFF14162B);
  static const Color inkSoft = Color(0xFF6B7280);

  static const Color primary = Color(0xFF2F6BFF); // Signal Blue
  static const Color primaryDeep = Color(0xFF1E40AF);
  static const Color secondary = Color(0xFF8B5CF6); // Violet
  static const Color secondaryDeep = Color(0xFF6D28D9);
  static const Color success = Color(0xFF10B981); // Green
  static const Color successDeep = Color(0xFF047857);
  static const Color warning = Color(0xFFF59E0B); // Amber
  static const Color error = Color(0xFFF43F5E); // Coral
  static const Color indigo = Color(0xFF4F46E5); // Deep blue-violet
  static const Color indigoDeep = Color(0xFF3730A3);

  static const LinearGradient primaryGradient = LinearGradient(
    colors: [primary, secondary],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
}

/// Global ThemeData - wire this into your MaterialApp with
/// `theme: AppTheme.light`. Once applied, Card / FilledButton / AppBar /
/// Chip / TextFormField across the whole module pick up the brand palette
/// automatically, with no per-widget edits required.
class AppTheme {
  AppTheme._();

  static ThemeData get light {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.light,
      primary: AppColors.primary,
      secondary: AppColors.secondary,
      tertiary: AppColors.success,
      error: AppColors.error,
      surface: AppColors.background,
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: AppColors.background,
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.background,
        foregroundColor: AppColors.ink,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: AppColors.ink,
          fontSize: 20,
          fontWeight: FontWeight.w700,
        ),
        iconTheme: IconThemeData(color: AppColors.ink),
      ),
      cardTheme: CardThemeData(
        color: Colors.white,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
          side: const BorderSide(color: AppColors.hairline),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          textStyle: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(foregroundColor: AppColors.primary),
      ),
      chipTheme: ChipThemeData(
        backgroundColor: AppColors.primary.withOpacity(0.08),
        labelStyle: const TextStyle(
          color: AppColors.primary,
          fontWeight: FontWeight.w600,
          fontSize: 12,
        ),
        side: BorderSide(color: AppColors.primary.withOpacity(0.2)),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.primary, width: 1.5),
        ),
        labelStyle: const TextStyle(color: AppColors.inkSoft),
      ),
      listTileTheme: const ListTileThemeData(iconColor: AppColors.primary),
      dividerTheme: const DividerThemeData(color: AppColors.hairline),
      textTheme: const TextTheme(
        headlineMedium:
            TextStyle(color: AppColors.ink, fontWeight: FontWeight.w800),
        headlineSmall:
            TextStyle(color: AppColors.ink, fontWeight: FontWeight.w800),
        titleLarge:
            TextStyle(color: AppColors.ink, fontWeight: FontWeight.w700),
        titleMedium:
            TextStyle(color: AppColors.ink, fontWeight: FontWeight.w700),
        bodyMedium: TextStyle(color: AppColors.inkSoft),
        bodyLarge: TextStyle(color: AppColors.inkSoft),
      ),
    );
  }
}
