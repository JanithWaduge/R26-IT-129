import 'package:flutter/material.dart';
import 'home_screen_janith.dart';

class LandingScreen extends StatefulWidget {
  const LandingScreen({super.key});

  @override
  State<LandingScreen> createState() => _LandingScreenState();
}

class _LandingScreenState extends State<LandingScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animController;
  late Animation<double>   _fadeAnim;
  late Animation<Offset>   _slideAnim;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 900));
    _fadeAnim  = CurvedAnimation(parent: _animController, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(begin: const Offset(0, 0.08), end: Offset.zero)
        .animate(CurvedAnimation(parent: _animController, curve: Curves.easeOut));
    _animController.forward();
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF020818),
      body: Stack(
        children: [
          Positioned(top: -80, left: -60,  child: _glow(280, const Color(0xFF00B4D8), 0.08)),
          Positioned(top: 240, right: -80, child: _glow(220, const Color(0xFF7B2FBE), 0.06)),
          Positioned(bottom: -60, left: -40, child: _glow(200, const Color(0xFF06D6A0), 0.05)),
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

  Widget _glow(double size, Color color, double opacity) => Container(
        width: size, height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          boxShadow: [BoxShadow(
            color: color.withOpacity(opacity),
            blurRadius: size, spreadRadius: size * 0.5,
          )],
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
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
            decoration: BoxDecoration(
              border      : Border.all(color: const Color(0xFF00B4D8).withOpacity(0.4)),
              borderRadius: BorderRadius.circular(30),
              color       : const Color(0xFF00B4D8).withOpacity(0.07),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Container(width: 6, height: 6,
                  decoration: const BoxDecoration(shape: BoxShape.circle, color: Color(0xFF00B4D8))),
              const SizedBox(width: 8),
              const Text('R26-IT-129  •  SLIIT Research  •  2026',
                  style: TextStyle(color: Color(0xFF90E0EF), fontSize: 11,
                      fontWeight: FontWeight.w600, letterSpacing: 0.6)),
            ]),
          ),
          const SizedBox(height: 22),
          RichText(
            text: const TextSpan(
              style: TextStyle(fontFamily: 'Roboto', fontSize: 34,
                  fontWeight: FontWeight.w800, height: 1.15, letterSpacing: -0.8),
              children: [
                TextSpan(text: 'Communicate\n', style: TextStyle(color: Colors.white)),
                TextSpan(text: 'Without\n',     style: TextStyle(color: Colors.white)),
                TextSpan(text: 'Barriers.',
                    style: TextStyle(color: Color(0xFF00B4D8), fontStyle: FontStyle.italic)),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'A bidirectional mobile application for Sinhala Sign Language '
            '— bringing real-time communication and learning to the '
            'Deaf and Hard of Hearing community in Sri Lanka.',
            style: TextStyle(color: Colors.white.withOpacity(0.55), fontSize: 13, height: 1.6),
          ),
          const SizedBox(height: 32),
          SizedBox(
            width: double.infinity, height: 56,
            child: ElevatedButton(
              onPressed: () => Navigator.push(context,
                  MaterialPageRoute(builder: (_) => const HomeScreen())),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF00B4D8),
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
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
          const SizedBox(height: 36),
          _buildStatsRow(),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _buildStatsRow() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 4),
      decoration: BoxDecoration(
        border: Border(
          top   : BorderSide(color: Colors.white.withOpacity(0.07)),
          bottom: BorderSide(color: Colors.white.withOpacity(0.07)),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _statItem('4', 'AI\nModules'),
          _statDivider(),
          _statItem('30', 'Classroom\nSigns'),
          _statDivider(),
          _statItem('Real\nTime', 'Live\nDetection'),
        ],
      ),
    );
  }

  Widget _statItem(String value, String label) => Column(children: [
        Text(value, textAlign: TextAlign.center, style: const TextStyle(
            color: Color(0xFF00B4D8), fontSize: 22, fontWeight: FontWeight.w800, height: 1.1)),
        const SizedBox(height: 4),
        Text(label, textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 10, height: 1.4)),
      ]);

  Widget _statDivider() =>
      Container(height: 36, width: 1, color: Colors.white10,
          margin: const EdgeInsets.symmetric(horizontal: 8));

  // ════════════════════════════════════════════
  // HOW GESTURE RECOGNITION WORKS
  // ════════════════════════════════════════════
  Widget _buildFeatureSection() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 36),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel('HOW GESTURE RECOGNITION WORKS'),
          const SizedBox(height: 20),
          SizedBox(
            height: 168,
            child: ListView(
              scrollDirection: Axis.horizontal,
              physics        : const BouncingScrollPhysics(),
              children: [
                _stepCard('01', Icons.camera_alt_outlined,
                    'Point Camera', 'Aim at the signer\'s hands in good lighting.', const Color(0xFF00B4D8)),
                const SizedBox(width: 12),
                _stepCard('02', Icons.back_hand_outlined,
                    'Hold the Sign', 'Keep the sign steady for about 3 seconds.', const Color(0xFF7B2FBE)),
                const SizedBox(width: 12),
                _stepCard('03', Icons.filter_alt_outlined,
                    'Noise Filter', 'Velocity filter removes accidental movements.', const Color(0xFFEF476F)),
                const SizedBox(width: 12),
                _stepCard('04', Icons.translate_rounded,
                    'Get Translation', 'See the sign in English and Sinhala.', const Color(0xFF06D6A0)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _stepCard(String step, IconData icon, String title, String body, Color color) {
    return Container(
      width: 180, padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.07), borderRadius: BorderRadius.circular(18),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Text(step, style: TextStyle(color: color.withOpacity(0.4),
              fontSize: 11, fontWeight: FontWeight.w800, letterSpacing: 1)),
          const Spacer(),
          Icon(icon, color: color, size: 20),
        ]),
        const Spacer(),
        Text(title, style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.w700)),
        const SizedBox(height: 6),
        Text(body, style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 11, height: 1.4),
            maxLines: 3, overflow: TextOverflow.ellipsis),
      ]),
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
          _sectionLabel('FOUR INTEGRATED AI MODULES'),
          const SizedBox(height: 20),

          // 02 — Hansamana (Available)
          _ModuleCard(
            moduleNo   : '02',
            icon       : Icons.sign_language,
            title      : 'Real-Time Gesture Recognition',
            description: 'Detects 30 classroom SLSL gestures using MediaPipe keypoints '
                'and CNN+LSTM with a novel velocity-threshold noise filter.',
            tags       : ['MediaPipe', 'CNN + LSTM', 'Noise Filter'],
            accentColor: const Color(0xFF00B4D8),
            isAvailable: true,
            onTap      : () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => const HomeScreen())),
          ),
          const SizedBox(height: 14),

          // 01 — Gimhana
          _ModuleCard(
            moduleNo   : '01',
            icon       : Icons.record_voice_over_rounded,
            title      : 'Voice / Text to Sign Language',
            description: 'Converts Sinhala & Tamil voice or text into 3D animated '
                'sign language following SLSL sentence structure.',
            tags       : ['ASR Whisper', '3D Avatar', 'Bilingual'],
            accentColor: const Color(0xFF7B2FBE),
            isAvailable: false,
            onTap      : () => _showUnavailable(context),
          ),
          const SizedBox(height: 14),

          // 03 — Dulmin
          _ModuleCard(
            moduleNo   : '03',
            icon       : Icons.school_rounded,
            title      : 'Adaptive Lesson System',
            description: 'Personalized lessons with placement testing — weak signs '
                'repeat automatically, difficulty adapts to performance.',
            tags       : ['Adaptive', 'Quizzes', 'Progress'],
            accentColor: const Color(0xFF06D6A0),
            isAvailable: false,
            onTap      : () => _showUnavailable(context),
          ),
          const SizedBox(height: 14),

          // 04 — Indumini
          _ModuleCard(
            moduleNo   : '04',
            icon       : Icons.dashboard_rounded,
            title      : 'Teacher Dashboard & Authoring',
            description: 'Lets teachers create and manage sign content and monitor '
                'student progress for the Sri Lankan curriculum.',
            tags       : ['Dashboard', 'Authoring', 'Analytics'],
            accentColor: const Color(0xFFFFB703),
            isAvailable: false,
            onTap      : () => _showUnavailable(context),
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
      margin : const EdgeInsets.fromLTRB(24, 0, 24, 40),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color       : Colors.white.withOpacity(0.03),
        borderRadius: BorderRadius.circular(20),
        border      : Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(children: [
        const Icon(Icons.accessibility_new_rounded, color: Color(0xFF00B4D8), size: 32),
        const SizedBox(height: 12),
        const Text('Built for Inclusion',
            style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
        const SizedBox(height: 8),
        Text('Empowering 400,000+ Deaf and Hard of Hearing individuals '
            'across Sri Lanka with accessible communication.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white.withOpacity(0.45), fontSize: 12, height: 1.6)),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: const Color(0xFF00B4D8).withOpacity(0.08),
            borderRadius: BorderRadius.circular(10),
          ),
          child: const Text('R26-IT-129  •  SLIIT  •  2026',
              style: TextStyle(color: Color(0xFF90E0EF), fontSize: 11,
                  fontWeight: FontWeight.w600, letterSpacing: 0.5)),
        ),
      ]),
    );
  }

  Widget _sectionLabel(String text) => Padding(
        padding: const EdgeInsets.only(bottom: 14),
        child: Text(text, style: TextStyle(
            color: Colors.white.withOpacity(0.35), fontSize: 11,
            fontWeight: FontWeight.w700, letterSpacing: 2)),
      );

  void _showUnavailable(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content        : const Text('This module is developed by another team member'),
      backgroundColor: const Color(0xFF023E8A),
      behavior       : SnackBarBehavior.floating,
      shape          : RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      duration       : const Duration(seconds: 2),
    ));
  }
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
    required this.icon, required this.title, required this.description,
    required this.tags, required this.accentColor,
    required this.isAvailable, required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color       : isAvailable ? accentColor.withOpacity(0.06) : Colors.white.withOpacity(0.02),
          borderRadius: BorderRadius.circular(20),
          border      : Border.all(
            color: isAvailable ? accentColor.withOpacity(0.3) : Colors.white.withOpacity(0.07),
            width: isAvailable ? 1.5 : 1,
          ),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Column(children: [
            Container(
              width: 52, height: 52,
              decoration: BoxDecoration(
                color       : accentColor.withOpacity(isAvailable ? 0.15 : 0.06),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon,
                  color: isAvailable ? accentColor : accentColor.withOpacity(0.4), size: 26),
            ),
            const SizedBox(height: 6),
            Text(moduleNo, style: TextStyle(
                color: isAvailable ? accentColor.withOpacity(0.6) : Colors.white24,
                fontSize: 11, fontWeight: FontWeight.w800, letterSpacing: 1)),
          ]),
          const SizedBox(width: 14),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Expanded(child: Text(title, style: TextStyle(
                  color: isAvailable ? Colors.white : Colors.white.withOpacity(0.5),
                  fontSize: 14, fontWeight: FontWeight.w700, height: 1.2))),
              const SizedBox(width: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: isAvailable ? accentColor.withOpacity(0.15) : Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(isAvailable ? 'Available' : 'Module',
                    style: TextStyle(
                        color: isAvailable ? accentColor : Colors.white30,
                        fontSize: 9, fontWeight: FontWeight.w700)),
              ),
            ]),
            const SizedBox(height: 6),
            Text(description, style: TextStyle(
                color  : Colors.white.withOpacity(isAvailable ? 0.55 : 0.35),
                fontSize: 12, height: 1.45),
                maxLines: 3, overflow: TextOverflow.ellipsis),
            const SizedBox(height: 10),
            Wrap(spacing: 6, runSpacing: 6, children: tags.map((tag) => Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: accentColor.withOpacity(isAvailable ? 0.1 : 0.04),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(tag, style: TextStyle(
                      color: isAvailable ? accentColor : accentColor.withOpacity(0.35),
                      fontSize: 10, fontWeight: FontWeight.w600)),
                )).toList()),
          ])),
          const SizedBox(width: 6),
          Padding(
            padding: const EdgeInsets.only(top: 18),
            child: Icon(Icons.arrow_forward_ios_rounded,
                size: 14, color: isAvailable ? accentColor.withOpacity(0.7) : Colors.white12),
          ),
        ]),
      ),
    );
  }
}