import 'package:flutter/material.dart';
import 'home_screen.dart';

// ══════════════════════════════════════════════════════════════════
// Landing Screen
// Professional, user-facing — no researcher names shown
// Any user can understand and navigate this immediately
// ══════════════════════════════════════════════════════════════════

class LandingScreen extends StatefulWidget {
  const LandingScreen({super.key});

  @override
  State<LandingScreen> createState() => _LandingScreenState();
}

class _LandingScreenState extends State<LandingScreen>
    with TickerProviderStateMixin {
  late AnimationController _heroController;
  late AnimationController _cardsController;
  late Animation<double>   _heroFade;
  late Animation<Offset>   _heroSlide;
  late Animation<double>   _cardsFade;

  @override
  void initState() {
    super.initState();

    _heroController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 800));
    _cardsController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 600));

    _heroFade  = CurvedAnimation(parent: _heroController,  curve: Curves.easeOut);
    _heroSlide = Tween<Offset>(begin: const Offset(0, 0.06), end: Offset.zero)
        .animate(CurvedAnimation(parent: _heroController, curve: Curves.easeOut));
    _cardsFade = CurvedAnimation(parent: _cardsController, curve: Curves.easeOut);

    _heroController.forward().then((_) {
      Future.delayed(const Duration(milliseconds: 100), () {
        if (mounted) _cardsController.forward();
      });
    });
  }

  @override
  void dispose() {
    _heroController.dispose();
    _cardsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: const Color(0xFF020818),
      body: Stack(
        children: [
          // Background glow effects
          Positioned(
            top: -80, left: -60,
            child: _glowCircle(280, const Color(0xFF00B4D8), 0.08),
          ),
          Positioned(
            top: 200, right: -80,
            child: _glowCircle(200, const Color(0xFF7B2FBE), 0.06),
          ),
          Positioned(
            bottom: 100, left: size.width * 0.3,
            child: _glowCircle(160, const Color(0xFF00B4D8), 0.05),
          ),

          // Main content
          SafeArea(
            child: CustomScrollView(
              physics: const BouncingScrollPhysics(),
              slivers: [
                SliverToBoxAdapter(
                  child: FadeTransition(
                    opacity: _heroFade,
                    child: SlideTransition(
                      position: _heroSlide,
                      child: _buildHero(),
                    ),
                  ),
                ),
                SliverToBoxAdapter(
                  child: FadeTransition(
                    opacity: _cardsFade,
                    child: _buildFeatureSection(),
                  ),
                ),
                SliverToBoxAdapter(
                  child: FadeTransition(
                    opacity: _cardsFade,
                    child: _buildModulesSection(),
                  ),
                ),
                SliverToBoxAdapter(
                  child: FadeTransition(
                    opacity: _cardsFade,
                    child: _buildFooter(),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Background glow ──────────────────────────
  Widget _glowCircle(double size, Color color, double opacity) {
    return Container(
      width : size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color     : color.withOpacity(opacity),
            blurRadius: size,
            spreadRadius: size * 0.5,
          ),
        ],
      ),
    );
  }

  // ── Hero ─────────────────────────────────────
  Widget _buildHero() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 32, 24, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Top pill badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
            decoration: BoxDecoration(
              border      : Border.all(color: const Color(0xFF00B4D8).withOpacity(0.4)),
              borderRadius: BorderRadius.circular(30),
              color       : const Color(0xFF00B4D8).withOpacity(0.07),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width : 6, height: 6,
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    color: Color(0xFF00B4D8),
                  ),
                ),
                const SizedBox(width: 8),
                const Text(
                  'Sri Lanka Sign Language  •  2026',
                  style: TextStyle(
                    color    : Color(0xFF90E0EF),
                    fontSize : 11,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.6,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 22),

          // Main headline
          RichText(
            text: const TextSpan(
              style: TextStyle(
                fontFamily: 'Roboto',
                fontSize  : 34,
                fontWeight: FontWeight.w800,
                height    : 1.15,
                letterSpacing: -0.8,
              ),
              children: [
                TextSpan(
                  text : 'Communicate\n',
                  style: TextStyle(color: Colors.white),
                ),
                TextSpan(
                  text : 'Without\n',
                  style: TextStyle(color: Colors.white),
                ),
                TextSpan(
                  text : 'Barriers.',
                  style: TextStyle(
                    color: Color(0xFF00B4D8),
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          Text(
            'A real-time bidirectional communication app for the '
            'Deaf and Hard of Hearing community — bridging '
            'Sri Lanka Sign Language with everyday conversation.',
            style: TextStyle(
              color : Colors.white.withOpacity(0.55),
              fontSize: 14,
              height: 1.6,
            ),
          ),
          const SizedBox(height: 32),

          // Primary CTA button
          SizedBox(
            width : double.infinity,
            height: 56,
            child: ElevatedButton(
              onPressed: () => Navigator.push(context,
                  MaterialPageRoute(builder: (_) => const HomeScreen())),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF00B4D8),
                foregroundColor: Colors.white,
                elevation      : 0,
                shape          : RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16)),
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.sign_language, size: 22),
                  SizedBox(width: 10),
                  Text(
                    'Start Sign Recognition',
                    style: TextStyle(
                        fontSize  : 16,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.3),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Secondary CTA
          SizedBox(
            width : double.infinity,
            height: 50,
            child: OutlinedButton(
              onPressed: () => _scrollToModules(),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.white70,
                side           : const BorderSide(color: Colors.white12),
                shape          : RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16)),
              ),
              child: const Text(
                'Explore Features',
                style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
              ),
            ),
          ),
          const SizedBox(height: 36),

          // Quick stats
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
          _statItem('30', 'Signs\nSupported'),
          _statDivider(),
          _statItem('77%+', 'Recognition\nAccuracy'),
          _statDivider(),
          _statItem('Real\nTime', 'Live\nDetection'),
        ],
      ),
    );
  }

  Widget _statItem(String value, String label) {
    return Column(
      children: [
        Text(value,
            style: const TextStyle(
                color     : Color(0xFF00B4D8),
                fontSize  : 22,
                fontWeight: FontWeight.w800,
                letterSpacing: -0.5)),
        const SizedBox(height: 4),
        Text(label,
            textAlign: TextAlign.center,
            style: TextStyle(
                color  : Colors.white.withOpacity(0.4),
                fontSize: 10,
                height : 1.4)),
      ],
    );
  }

  Widget _statDivider() =>
      Container(height: 36, width: 1, color: Colors.white10);

  // ── Feature highlights ───────────────────────
  Widget _buildFeatureSection() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 36),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel('HOW IT WORKS'),
          const SizedBox(height: 20),

          // Step cards — horizontal scroll
          SizedBox(
            height: 160,
            child: ListView(
              scrollDirection: Axis.horizontal,
              physics        : const BouncingScrollPhysics(),
              children: [
                _stepCard(
                  step    : '01',
                  icon    : Icons.camera_alt_outlined,
                  title   : 'Point Camera',
                  body    : 'Aim your camera at the signer\'s hands in good lighting.',
                  color   : const Color(0xFF00B4D8),
                ),
                const SizedBox(width: 12),
                _stepCard(
                  step    : '02',
                  icon    : Icons.back_hand_outlined,
                  title   : 'Hold the Sign',
                  body    : 'Keep the sign steady for about 3 seconds while the app captures it.',
                  color   : const Color(0xFF7B2FBE),
                ),
                const SizedBox(width: 12),
                _stepCard(
                  step    : '03',
                  icon    : Icons.translate_rounded,
                  title   : 'Get Translation',
                  body    : 'See the sign name in English and Sinhala with confidence score.',
                  color   : const Color(0xFF06D6A0),
                ),
                const SizedBox(width: 12),
                _stepCard(
                  step    : '04',
                  icon    : Icons.bar_chart_rounded,
                  title   : 'Check Accuracy',
                  body    : 'View confidence percentage and top alternative signs detected.',
                  color   : const Color(0xFFFFB703),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _stepCard({
    required String   step,
    required IconData icon,
    required String   title,
    required String   body,
    required Color    color,
  }) {
    return Container(
      width  : 180,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color       : color.withOpacity(0.07),
        borderRadius: BorderRadius.circular(18),
        border      : Border.all(color: color.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(step,
                  style: TextStyle(
                      color    : color.withOpacity(0.4),
                      fontSize : 11,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1)),
              const Spacer(),
              Icon(icon, color: color, size: 20),
            ],
          ),
          const Spacer(),
          Text(title,
              style: TextStyle(
                  color     : color,
                  fontSize  : 14,
                  fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(body,
              style: TextStyle(
                  color  : Colors.white.withOpacity(0.5),
                  fontSize: 11,
                  height : 1.4),
              maxLines: 3,
              overflow: TextOverflow.ellipsis),
        ],
      ),
    );
  }

  // ── Modules section ──────────────────────────
  Widget _buildModulesSection() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 36),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel('APP MODULES'),
          const SizedBox(height: 20),

          // Sign Recognition — YOUR module (active)
          _ModuleCard(
            icon       : Icons.sign_language,
            title      : 'Sign Language Recognition',
            description: 'Real-time detection of 30 classroom Sri Lanka Sign Language '
                'gestures using AI with noise filtering technology.',
            tags       : ['AI Powered', '30 Signs', 'CNN + LSTM'],
            accentColor: const Color(0xFF00B4D8),
            isAvailable: true,
            onTap      : () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => const HomeScreen())),
          ),
          const SizedBox(height: 14),

          // Module 2 placeholder
          _ModuleCard(
            icon       : Icons.record_voice_over_rounded,
            title      : 'Text to Sign Language',
            description: 'Convert spoken or typed text into sign language animations '
                'for seamless communication.',
            tags       : ['Text Input', 'Animation', 'Bilingual'],
            accentColor: const Color(0xFF7B2FBE),
            isAvailable: false,
            onTap      : () => _showUnavailable(context),
          ),
          const SizedBox(height: 14),

          // Module 3 placeholder
          _ModuleCard(
            icon       : Icons.hearing_rounded,
            title      : 'Speech to Text',
            description: 'Convert spoken language to text in real-time to help DHH '
                'users follow conversations.',
            tags       : ['Voice Input', 'Real-time', 'Sinhala'],
            accentColor: const Color(0xFF06D6A0),
            isAvailable: false,
            onTap      : () => _showUnavailable(context),
          ),
          const SizedBox(height: 14),

          // Module 4 placeholder
          _ModuleCard(
            icon       : Icons.forum_rounded,
            title      : 'Community & Learning',
            description: 'Practice sign language with interactive lessons and connect '
                'with the DHH community.',
            tags       : ['Lessons', 'Community', 'Progress'],
            accentColor: const Color(0xFFFFB703),
            isAvailable: false,
            onTap      : () => _showUnavailable(context),
          ),
        ],
      ),
    );
  }

  // ── Footer ───────────────────────────────────
  Widget _buildFooter() {
    return Container(
      margin : const EdgeInsets.fromLTRB(24, 0, 24, 40),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color       : Colors.white.withOpacity(0.03),
        borderRadius: BorderRadius.circular(20),
        border      : Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        children: [
          const Icon(Icons.accessibility_new_rounded,
              color: Color(0xFF00B4D8), size: 32),
          const SizedBox(height: 12),
          const Text(
            'Built for Inclusion',
            style: TextStyle(
                color     : Colors.white,
                fontSize  : 16,
                fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          Text(
            'This application is designed to empower the Deaf and '
            'Hard of Hearing community by enabling effortless '
            'communication in everyday settings.',
            textAlign: TextAlign.center,
            style: TextStyle(
                color  : Colors.white.withOpacity(0.45),
                fontSize: 12,
                height : 1.6),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color       : const Color(0xFF00B4D8).withOpacity(0.08),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Text(
              'SLIIT Research Project  •  2026',
              style: TextStyle(
                  color    : Color(0xFF90E0EF),
                  fontSize : 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionLabel(String text) {
    return Text(
      text,
      style: TextStyle(
        color    : Colors.white.withOpacity(0.35),
        fontSize : 11,
        fontWeight: FontWeight.w700,
        letterSpacing: 2,
      ),
    );
  }

  void _scrollToModules() {}

  void _showUnavailable(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('This module is coming soon'),
        backgroundColor: const Color(0xFF023E8A),
        behavior       : SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        duration: const Duration(seconds: 2),
      ),
    );
  }
}

// ══════════════════════════════════════════════
// Module Card
// ══════════════════════════════════════════════
class _ModuleCard extends StatelessWidget {
  final IconData     icon;
  final String       title;
  final String       description;
  final List<String> tags;
  final Color        accentColor;
  final bool         isAvailable;
  final VoidCallback onTap;

  const _ModuleCard({
    required this.icon,
    required this.title,
    required this.description,
    required this.tags,
    required this.accentColor,
    required this.isAvailable,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding : const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color       : isAvailable
              ? accentColor.withOpacity(0.06)
              : Colors.white.withOpacity(0.02),
          borderRadius: BorderRadius.circular(20),
          border      : Border.all(
            color: isAvailable
                ? accentColor.withOpacity(0.3)
                : Colors.white.withOpacity(0.07),
            width: isAvailable ? 1.5 : 1,
          ),
        ),
        child: Row(
          children: [
            // Icon box
            Container(
              width : 52,
              height: 52,
              decoration: BoxDecoration(
                color       : accentColor.withOpacity(isAvailable ? 0.15 : 0.06),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(
                icon,
                color: isAvailable ? accentColor : accentColor.withOpacity(0.4),
                size : 26,
              ),
            ),
            const SizedBox(width: 14),

            // Text content
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(title,
                            style: TextStyle(
                              color     : isAvailable
                                  ? Colors.white
                                  : Colors.white.withOpacity(0.45),
                              fontSize  : 14,
                              fontWeight: FontWeight.w700,
                            )),
                      ),
                      if (isAvailable)
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color       : accentColor.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text('Available',
                              style: TextStyle(
                                  color    : accentColor,
                                  fontSize : 9,
                                  fontWeight: FontWeight.w700)),
                        )
                      else
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color       : Colors.white.withOpacity(0.05),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Text('Soon',
                              style: TextStyle(
                                  color    : Colors.white30,
                                  fontSize : 9,
                                  fontWeight: FontWeight.w700)),
                        ),
                    ],
                  ),
                  const SizedBox(height: 5),
                  Text(description,
                      style: TextStyle(
                          color  : Colors.white.withOpacity(isAvailable ? 0.5 : 0.3),
                          fontSize: 12,
                          height : 1.4),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 6,
                    children: tags.map((tag) => Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color       : accentColor.withOpacity(
                                isAvailable ? 0.1 : 0.04),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(tag,
                              style: TextStyle(
                                  color  : isAvailable
                                      ? accentColor
                                      : accentColor.withOpacity(0.35),
                                  fontSize: 10,
                                  fontWeight: FontWeight.w600)),
                        )).toList(),
                  ),
                ],
              ),
            ),

            // Arrow
            Padding(
              padding: const EdgeInsets.only(left: 8),
              child: Icon(
                Icons.arrow_forward_ios_rounded,
                size : 14,
                color: isAvailable
                    ? accentColor.withOpacity(0.7)
                    : Colors.white12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}