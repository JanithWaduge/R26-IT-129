import 'package:flutter/material.dart';
import 'home_screen_janith.dart';
import 'teacher_dashboard_screen_hansika.dart';
import '../bams/features/authentication/presentation/auth_gate.dart';

class LandingScreen extends StatefulWidget {
  const LandingScreen({super.key});

  @override
  State<LandingScreen> createState() => _LandingScreenState();
}

class _LandingScreenState extends State<LandingScreen>
    with TickerProviderStateMixin {
  late AnimationController _animController;
  late Animation<double> _fadeAnim;
  late Animation<Offset> _slideAnim;

  // Signature "detection pulse" — a looping animation used in the hero to
  // echo the product's core idea: a sign being scanned and recognized live.
  late AnimationController _pulseController;

  static const _white = Color(0xFFFFFFFF);
  static const _ink = Color(0xFF14162B);
  static const _inkSoft = Color(0xFF6B7280);

  static const _cyan = Color(0xFF2F6BFF);
  static const _violet = Color(0xFF8B5CF6);
  static const _emerald = Color(0xFF10B981);
  static const _amber = Color(0xFF4F46E5);

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 900));
    _fadeAnim =
        CurvedAnimation(parent: _animController, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(begin: const Offset(0, 0.08), end: Offset.zero)
        .animate(CurvedAnimation(parent: _animController, curve: Curves.easeOut));
    _animController.forward();

    _pulseController =
        AnimationController(vsync: this, duration: const Duration(seconds: 2))
          ..repeat();
  }

  @override
  void dispose() {
    _animController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _white,
      body: Stack(
        children: [
          Positioned(top: -70, left: -60, child: _blob(260, _cyan, 0.16)),
          Positioned(top: 220, right: -90, child: _blob(230, _violet, 0.14)),
          Positioned(bottom: -50, left: -50, child: _blob(220, _emerald, 0.13)),
          SafeArea(
            child: FadeTransition(
              opacity: _fadeAnim,
              child: SlideTransition(
                position: _slideAnim,
                child: CustomScrollView(
                  physics: const BouncingScrollPhysics(),
                  slivers: [
                    SliverToBoxAdapter(child: _buildHero()),
                    SliverToBoxAdapter(child: _buildFeatureSection()),
                    SliverToBoxAdapter(child: _buildModulesSection()),
                    SliverToBoxAdapter(child: _buildFooter()),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _blob(double size, Color color, double opacity) => Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: RadialGradient(colors: [
            color.withOpacity(opacity),
            color.withOpacity(0),
          ]),
        ),
      );

  // ════════════════════════════════════════════
  // HERO
  // ════════════════════════════════════════════
  Widget _buildHero() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 32, 24, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Eyebrow — replaces the old project-code pill with a live
          // "detection pulse" that mirrors the app's own core behaviour.
          Row(
            children: [
              SizedBox(
                width: 30,
                height: 30,
                child: AnimatedBuilder(
                  animation: _pulseController,
                  builder: (_, __) => CustomPaint(
                    painter: _PulsePainter(_pulseController.value, _cyan),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              const Text(
                'SIGN LANGUAGE, RECOGNIZED LIVE',
                style: TextStyle(
                  color: Color(0xFF1E40AF),
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.4,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          RichText(
            text: const TextSpan(
              style: TextStyle(
                fontFamily: 'Roboto',
                fontSize: 36,
                fontWeight: FontWeight.w800,
                height: 1.12,
                letterSpacing: -0.9,
              ),
              children: [
                TextSpan(text: 'Communicate\n', style: TextStyle(color: Color(0xFF14162B))),
                TextSpan(text: 'Without\n', style: TextStyle(color: Color(0xFF14162B))),
                TextSpan(
                    text: 'Barriers.',
                    style: TextStyle(color: Color(0xFF2F6BFF), fontStyle: FontStyle.italic)),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'A bidirectional mobile app for Sri Lankan Sign Language — '
            'real-time recognition, translation, and teaching, built for the '
            'Deaf and Hard of Hearing community.',
            style: TextStyle(color: _inkSoft, fontSize: 13, height: 1.6),
          ),
          const SizedBox(height: 32),
          SizedBox(
            width: double.infinity,
            height: 58,
            child: DecoratedBox(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(18),
                gradient: const LinearGradient(
                  begin: Alignment.centerLeft,
                  end: Alignment.centerRight,
                  colors: [Color(0xFF2F6BFF), Color(0xFF8B5CF6)],
                ),
                boxShadow: [
                  BoxShadow(
                    color: _cyan.withOpacity(0.35),
                    blurRadius: 24,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: ElevatedButton(
                onPressed: () => Navigator.push(
                    context, MaterialPageRoute(builder: (_) => const HomeScreen())),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.transparent,
                  foregroundColor: Colors.white,
                  shadowColor: Colors.transparent,
                  elevation: 0,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                ),
                child: const Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.sign_language, size: 22),
                    SizedBox(width: 10),
                    Text('Start Sign Recognition',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 36),
          _buildStatsRow(),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _buildStatsRow() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 12),
      decoration: BoxDecoration(
        color: _white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFEDEFF5)),
        boxShadow: [
          BoxShadow(
            color: _ink.withOpacity(0.05),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _statItem(Icons.auto_awesome_rounded, '4', 'AI\nModules', _cyan),
          _statDivider(),
          _statItem(Icons.front_hand_rounded, '30', 'Classroom\nSigns', _violet),
          _statDivider(),
          _statItem(Icons.bolt_rounded, 'Live', 'Real-Time\nDetection', _emerald),
        ],
      ),
    );
  }

  Widget _statItem(IconData icon, String value, String label, Color color) => Column(
        children: [
        Container(
        width: 34,
        height: 34,
        decoration: BoxDecoration(color: color.withOpacity(0.12), shape: BoxShape.circle),
        child: Icon(icon, color: color, size: 17),
        ),
          const SizedBox(height: 6),
          Text(value,
              textAlign: TextAlign.center,
          style: TextStyle(
            color: _ink, fontSize: 20, fontWeight: FontWeight.w800, height: 1.1)),
          const SizedBox(height: 4),
          Text(label,
              textAlign: TextAlign.center,
          style: TextStyle(color: _inkSoft, fontSize: 10, height: 1.4)),
        ],
      );

    Widget _statDivider() => Container(
      height: 40, width: 1, color: const Color(0xFFEDEFF5), margin: const EdgeInsets.symmetric(horizontal: 8));

  // ════════════════════════════════════════════
  // HOW GESTURE RECOGNITION WORKS
  // ════════════════════════════════════════════
  Widget _buildFeatureSection() {
    final steps = <_StepData>[
      _StepData('01', Icons.camera_alt_outlined, 'Point Camera',
          'Aim at the signer\'s hands in good lighting.', _cyan),
      _StepData('02', Icons.back_hand_outlined, 'Hold the Sign',
          'Keep the sign steady for about 3 seconds.', _violet),
      _StepData('03', Icons.filter_alt_outlined, 'Noise Filter',
          'Velocity filter removes accidental movements.', const Color(0xFFEF476F)),
      _StepData('04', Icons.translate_rounded, 'Get Translation',
          'See the sign in English and Sinhala.', _emerald),
    ];

    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 0, 0, 36),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(right: 24),
            child: _sectionLabel('HOW GESTURE RECOGNITION WORKS'),
          ),
          const SizedBox(height: 20),
          // Fixed, content-hugging height — the old 172px box combined with
          // two Spacer()s left a large empty gap when the text was short.
          SizedBox(
            height: 142,
            child: ListView.separated(
              padding: const EdgeInsets.only(right: 24),
              scrollDirection: Axis.horizontal,
              physics: const BouncingScrollPhysics(),
              itemCount: steps.length,
              separatorBuilder: (_, __) => Padding(
                padding: const EdgeInsets.symmetric(horizontal: 4),
                child: Icon(Icons.arrow_forward_rounded, color: _inkSoft.withOpacity(0.3), size: 16),
              ),
              itemBuilder: (_, i) {
                final s = steps[i];
                return _stepCard(s.step, s.icon, s.title, s.body, s.color);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _stepCard(String step, IconData icon, String title, String body, Color color) {
    return Container(
      width: 152,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFEDEFF5)),
        boxShadow: [
          BoxShadow(color: color.withOpacity(0.10), blurRadius: 18, offset: const Offset(0, 8)),
        ],
      ),
      // No Spacer() here — a Column with mainAxisSize.min sized to its
      // content keeps the card compact instead of stretching to fill height.
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(7),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: color, size: 16),
              ),
              const Spacer(),
              Text(step,
                  style: TextStyle(
                      color: color.withOpacity(0.4),
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1)),
            ],
          ),
          const SizedBox(height: 14),
          Text(title, style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(body,
              style: TextStyle(color: _inkSoft, fontSize: 11, height: 1.4),
              maxLines: 3,
              overflow: TextOverflow.ellipsis),
        ],
      ),
    );
  }

  // ════════════════════════════════════════════
  // FOUR INTEGRATED MODULES
  // ════════════════════════════════════════════
  Widget _buildModulesSection() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 36),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel('THE FOUR-PART SLSL AI SUITE'),
          const SizedBox(height: 20),

          // 02 — Hansamana (Available)
          _ModuleCard(
            moduleNo: '02',
            icon: Icons.sign_language,
            title: 'Real-Time Gesture Recognition',
            description: 'Detects 30 classroom SLSL gestures using MediaPipe keypoints '
                'and CNN+LSTM with a novel velocity-threshold noise filter.',
            tags: const ['MediaPipe', 'CNN + LSTM', 'Noise Filter'],
            accentColor: _cyan,
            isAvailable: true,
            onTap: () =>
                Navigator.push(context, MaterialPageRoute(builder: (_) => const HomeScreen())),
          ),
          const SizedBox(height: 14),

          // 01 — Gimhana
          _ModuleCard(
            moduleNo: '01',
            icon: Icons.record_voice_over_rounded,
            title: 'Voice / Text to Sign Language',
            description: 'Converts Sinhala & Tamil voice or text into 3D animated '
                'sign language following SLSL sentence structure.',
            tags: const ['ASR Whisper', '3D Avatar', 'Bilingual'],
            accentColor: _violet,
            isAvailable: false,
            onTap: () => _showUnavailable(context),
          ),
          const SizedBox(height: 14),

          // 03 — Kisal (Adaptive Lesson System — now Live, BAMS version)
          _ModuleCard(
            moduleNo: '03',
            icon: Icons.school_rounded,
            title: 'Adaptive Lesson System',
            description: 'Personalized lessons with placement testing — weak signs '
                'repeat automatically, difficulty adapts to performance.',
            tags: const ['Adaptive', 'Quizzes', 'Progress'],
            accentColor: _emerald,
            isAvailable: true,
            onTap: () => Navigator.push(
                context, MaterialPageRoute(builder: (_) => const AuthGate())),
          ),
          const SizedBox(height: 14),

          // 04 — Indumini (wired to Hansika's Teacher Dashboard implementation)
          _ModuleCard(
            moduleNo: '04',
            icon: Icons.dashboard_rounded,
            title: 'Teacher Dashboard & Authoring',
            description: 'Lets teachers create and manage sign content and monitor '
                'student progress for the Sri Lankan curriculum.',
            tags: const ['Dashboard', 'Authoring', 'Analytics'],
            accentColor: _amber,
            isAvailable: true,
            onTap: () => Navigator.push(
                context, MaterialPageRoute(builder: (_) => const TeacherDashboardScreenHansika())),
          ),
        ],
      ),
    );
  }

  // ════════════════════════════════════════════
  // FOOTER
  // ════════════════════════════════════════════
  Widget _buildFooter() {
    return Container(
      margin: const EdgeInsets.fromLTRB(24, 0, 24, 40),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFEDEFF5)),
        boxShadow: [
          BoxShadow(color: _ink.withOpacity(0.05), blurRadius: 22, offset: const Offset(0, 10)),
        ],
      ),
      child: Column(
        children: [
          const Icon(Icons.accessibility_new_rounded, color: Color(0xFF2F6BFF), size: 32),
          const SizedBox(height: 12),
          Text('Built for Inclusion',
              style: TextStyle(color: _ink, fontSize: 16, fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          Text(
            'Empowering 400,000+ Deaf and Hard of Hearing individuals '
            'across Sri Lanka with accessible communication.',
            textAlign: TextAlign.center,
            style: TextStyle(color: _inkSoft, fontSize: 12, height: 1.6),
          ),
          const SizedBox(height: 18),
          // Four dots — one per module — a quiet closing signature instead
          // of a project-code label.
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _footerDot(_violet),
              const SizedBox(width: 8),
              _footerDot(_cyan),
              const SizedBox(width: 8),
              _footerDot(_emerald),
              const SizedBox(width: 8),
              _footerDot(_amber),
            ],
          ),
        ],
      ),
    );
  }

  Widget _footerDot(Color color) => Container(
        width: 7,
        height: 7,
      decoration: BoxDecoration(shape: BoxShape.circle, color: color),
      );

  Widget _sectionLabel(String text) => Padding(
        padding: const EdgeInsets.only(bottom: 14),
        child: Text(
          text,
          style: TextStyle(
              color: _inkSoft,
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 2),
        ),
      );

  void _showUnavailable(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: const Text('This module is developed by another team member'),
      backgroundColor: _ink,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      duration: const Duration(seconds: 2),
    ));
  }
}

// ════════════════════════════════════════════
// STEP DATA — small helper for the "how it works" strip
// ════════════════════════════════════════════
class _StepData {
  final String step;
  final IconData icon;
  final String title;
  final String body;
  final Color color;
  const _StepData(this.step, this.icon, this.title, this.body, this.color);
}

// ════════════════════════════════════════════
// SIGNATURE PULSE PAINTER — used once, in the hero
// ════════════════════════════════════════════
class _PulsePainter extends CustomPainter {
  final double progress; // 0..1, looping
  final Color color;
  _PulsePainter(this.progress, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final maxRadius = size.width / 2;

    for (int i = 0; i < 3; i++) {
      final t = (progress + i / 3) % 1.0;
      final radius = maxRadius * t;
      final opacity = (1 - t) * 0.55;
      final paint = Paint()
        ..color = color.withOpacity(opacity)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5;
      canvas.drawCircle(center, radius, paint);
    }

    final corePaint = Paint()..color = color;
    canvas.drawCircle(center, maxRadius * 0.22, corePaint);
  }

  @override
  bool shouldRepaint(covariant _PulsePainter oldDelegate) => oldDelegate.progress != progress;
}

// ════════════════════════════════════════════
// MODULE CARD
// ════════════════════════════════════════════
class _ModuleCard extends StatelessWidget {
  final String moduleNo;
  final IconData icon;
  final String title, description;
  final List<String> tags;
  final Color accentColor;
  final bool isAvailable;
  final VoidCallback onTap;

  const _ModuleCard({
    required this.moduleNo,
    required this.icon,
    required this.title,
    required this.description,
    required this.tags,
    required this.accentColor,
    required this.isAvailable,
    required this.onTap,
  });

  static const _ink = Color(0xFF14162B);
  static const _inkSoft = Color(0xFF6B7280);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(22),
          border: Border.all(
            color: isAvailable ? accentColor.withOpacity(0.28) : const Color(0xFFEDEFF5),
            width: isAvailable ? 1.4 : 1,
          ),
          boxShadow: [
            BoxShadow(
              color: (isAvailable ? accentColor : _ink).withOpacity(isAvailable ? 0.14 : 0.04),
              blurRadius: 22,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Column(
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    color: accentColor.withOpacity(isAvailable ? 0.16 : 0.08),
                    borderRadius: BorderRadius.circular(15),
                    boxShadow: isAvailable
                        ? [BoxShadow(color: accentColor.withOpacity(0.35), blurRadius: 14, offset: const Offset(0, 6))]
                        : null,
                  ),
                  child: Icon(icon,
                      color: isAvailable ? accentColor : _inkSoft.withOpacity(0.5), size: 26),
                ),
                const SizedBox(height: 6),
                Text(moduleNo,
                    style: TextStyle(
                        color: isAvailable ? accentColor : _inkSoft.withOpacity(0.5),
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1)),
              ],
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(title,
                            style: TextStyle(
                                color: isAvailable ? _ink : _inkSoft,
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                height: 1.2)),
                      ),
                      const SizedBox(width: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: isAvailable
                              ? accentColor.withOpacity(0.12)
                              : const Color(0xFFF3F4F8),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            if (isAvailable)
                              Container(
                                width: 5,
                                height: 5,
                                margin: const EdgeInsets.only(right: 5),
                                decoration:
                                    BoxDecoration(shape: BoxShape.circle, color: accentColor),
                              )
                            else
                              Icon(Icons.lock_outline_rounded, size: 10, color: _inkSoft.withOpacity(0.5)),
                            Text(isAvailable ? 'Live' : 'Locked',
                                style: TextStyle(
                                    color: isAvailable ? accentColor : _inkSoft.withOpacity(0.6),
                                    fontSize: 9,
                                    fontWeight: FontWeight.w700)),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(description,
                      style: TextStyle(
                        color: isAvailable ? _inkSoft : _inkSoft.withOpacity(0.6),
                          fontSize: 12,
                          height: 1.45),
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: tags
                        .map((tag) => Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                              decoration: BoxDecoration(
                                color: isAvailable ? accentColor.withOpacity(0.10) : const Color(0xFFF3F4F8),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: Text(tag,
                                  style: TextStyle(
                                      color: isAvailable
                                          ? accentColor
                                          : _inkSoft.withOpacity(0.5),
                                      fontSize: 10,
                                      fontWeight: FontWeight.w600)),
                            ))
                        .toList(),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 6),
            Padding(
              padding: const EdgeInsets.only(top: 18),
              child: Container(
                width: 26,
                height: 26,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isAvailable ? accentColor.withOpacity(0.12) : const Color(0xFFF3F4F8),
                ),
                child: Icon(Icons.arrow_forward_ios_rounded,
                    size: 12, color: isAvailable ? accentColor : _inkSoft.withOpacity(0.4)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}