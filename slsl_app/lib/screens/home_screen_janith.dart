import 'package:flutter/material.dart';
import '../constants.dart';
import 'camera_screen_janith.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnim;
  late Animation<Offset> _slideAnim;

  // ── Brand palette: white primary, blue / violet / green accents ──
  static const _ink = Color(0xFF14162B);
  static const _inkSoft = Color(0xFF6B7280);
  static const _blue = Color(0xFF2F6BFF);
  static const _blueDeep = Color(0xFF1E40AF);
  static const _violet = Color(0xFF8B5CF6);
  static const _green = Color(0xFF10B981);
  static const _hairline = Color(0xFFEDEFF5);

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 750));
    _fadeAnim = CurvedAnimation(parent: _controller, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(begin: const Offset(0, 0.05), end: Offset.zero)
        .animate(CurvedAnimation(parent: _controller, curve: Curves.easeOut));
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          Positioned(top: -40, right: -60, child: _blob(220, _blue, 0.16)),
          Positioned(bottom: 120, left: -50, child: _blob(200, _violet, 0.13)),
          SafeArea(
            child: FadeTransition(
              opacity: _fadeAnim,
              child: SlideTransition(
                position: _slideAnim,
                child: CustomScrollView(
                  physics: const BouncingScrollPhysics(),
                  slivers: [
                    SliverToBoxAdapter(child: _buildTopBar(context)),
                    SliverToBoxAdapter(child: _buildHeroCard()),
                    SliverToBoxAdapter(child: _buildStatsRow()),
                    SliverToBoxAdapter(child: _buildHowItWorksLabel()),
                    SliverToBoxAdapter(child: _buildStepsList()),
                    SliverToBoxAdapter(child: _buildStartButton(context)),
                    const SliverToBoxAdapter(child: SizedBox(height: 40)),
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
          gradient: RadialGradient(colors: [color.withOpacity(opacity), color.withOpacity(0)]),
        ),
      );

  Widget _buildTopBar(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 12, 20, 0),
      child: Row(children: [
        IconButton(
          icon: const Icon(Icons.arrow_back_ios_rounded, color: _inkSoft, size: 20),
          onPressed: () => Navigator.maybePop(context),
        ),
        const Spacer(),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            border: Border.all(color: _blue.withOpacity(0.25)),
            borderRadius: BorderRadius.circular(20),
            color: _blue.withOpacity(0.08),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Container(
                width: 6, height: 6, decoration: const BoxDecoration(shape: BoxShape.circle, color: _blue)),
            const SizedBox(width: 7),
            Text('AI Module',
                style: TextStyle(color: _blueDeep, fontSize: 11, fontWeight: FontWeight.w600)),
          ]),
        ),
      ]),
    );
  }

  Widget _buildHeroCard() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [_blue, _violet],
          ),
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
            BoxShadow(color: _blue.withOpacity(0.28), blurRadius: 30, offset: const Offset(0, 14)),
          ],
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.18),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(Icons.sign_language, color: Colors.white, size: 30),
            ),
            const SizedBox(width: 16),
            const Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                SizedBox(height: 2),
                Text('SLSL Recognizer',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -0.3,
                        height: 1.1)),
                SizedBox(height: 5),
                Text('Sri Lanka Sign Language',
                    style: TextStyle(color: Colors.white70, fontSize: 13, fontWeight: FontWeight.w500)),
              ]),
            ),
          ]),
          const SizedBox(height: 20),
          Container(height: 1, color: Colors.white.withOpacity(0.2)),
          const SizedBox(height: 18),
          Text(
            'Point your camera at a signer\'s hands and let the AI '
            'identify Sri Lanka Sign Language gestures in real time '
            'with bilingual Sinhala and English output.',
            style: TextStyle(color: Colors.white.withOpacity(0.85), fontSize: 13, height: 1.6),
          ),
        ]),
      ),
    );
  }

  Widget _buildStatsRow() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
      child: Row(children: [
        _statCard('${kSignLabels.length}', 'Signs', Icons.gesture_rounded, _blue),
        const SizedBox(width: 10),
        _statCard('30', 'Frames', Icons.video_camera_back_outlined, _violet),
        const SizedBox(width: 10),
        _statCard('CNN\nLSTM', 'Model', Icons.memory_rounded, _green),
      ]),
    );
  }

  Widget _statCard(String value, String label, IconData icon, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: _hairline),
          boxShadow: [
            BoxShadow(color: color.withOpacity(0.10), blurRadius: 16, offset: const Offset(0, 8)),
          ],
        ),
        child: Column(children: [
          Container(
            width: 30,
            height: 30,
            decoration: BoxDecoration(color: color.withOpacity(0.12), shape: BoxShape.circle),
            child: Icon(icon, color: color, size: 16),
          ),
          const SizedBox(height: 8),
          Text(value,
              textAlign: TextAlign.center,
              style: TextStyle(
                  color: _ink, fontSize: 16, fontWeight: FontWeight.w800, height: 1.15, letterSpacing: -0.3)),
          const SizedBox(height: 4),
          Text(label, style: TextStyle(color: _inkSoft, fontSize: 10, fontWeight: FontWeight.w500)),
        ]),
      ),
    );
  }

  Widget _buildHowItWorksLabel() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 16),
      child: Row(children: [
        Container(
            width: 3,
            height: 16,
            decoration: BoxDecoration(color: _blue, borderRadius: BorderRadius.circular(2))),
        const SizedBox(width: 10),
        Text('HOW IT WORKS',
            style: TextStyle(color: _inkSoft, fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 2)),
      ]),
    );
  }

  Widget _buildStepsList() {
    final steps = [
      _StepData('01', Icons.camera_alt_outlined, 'Point camera at hands',
          'Hold your sign clearly in frame with good lighting.', _blue),
      _StepData('02', Icons.auto_awesome_rounded, 'AI detects the sign',
          'CNN+LSTM model with velocity-based noise filtering.', _violet),
      _StepData('03', Icons.translate_rounded, 'See the result',
          'Sign name in English & Sinhala with confidence score.', _green),
    ];
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
          children: steps
              .asMap()
              .entries
              .map((e) => _buildStepTile(e.value, isLast: e.key == steps.length - 1))
              .toList()),
    );
  }

  Widget _buildStepTile(_StepData step, {required bool isLast}) {
    return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Column(children: [
        Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            gradient: LinearGradient(colors: [step.color, step.color.withOpacity(0.7)]),
            shape: BoxShape.circle,
          ),
          child: Center(
              child: Text(step.number,
                  style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.w800))),
        ),
        if (!isLast)
          Container(
              width: 1,
              height: 40,
              margin: const EdgeInsets.symmetric(vertical: 4),
              color: _hairline),
      ]),
      const SizedBox(width: 14),
      Expanded(
          child: Padding(
        padding: EdgeInsets.only(bottom: isLast ? 0 : 16, top: 6),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: _hairline),
            boxShadow: [
              BoxShadow(color: step.color.withOpacity(0.08), blurRadius: 14, offset: const Offset(0, 6)),
            ],
          ),
          child: Row(children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(color: step.color.withOpacity(0.12), borderRadius: BorderRadius.circular(11)),
              child: Icon(step.icon, color: step.color, size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(step.title,
                  style: const TextStyle(color: _ink, fontSize: 13, fontWeight: FontWeight.w700)),
              const SizedBox(height: 4),
              Text(step.body, style: TextStyle(color: _inkSoft, fontSize: 12, height: 1.4)),
            ])),
          ]),
        ),
      )),
    ]);
  }

  Widget _buildStartButton(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 0),
      child: Column(children: [
        SizedBox(
          width: double.infinity,
          height: 58,
          child: DecoratedBox(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(18),
              gradient: const LinearGradient(colors: [_blue, _violet]),
              boxShadow: [
                BoxShadow(color: _blue.withOpacity(0.32), blurRadius: 22, offset: const Offset(0, 10)),
              ],
            ),
            child: ElevatedButton(
              onPressed: () => Navigator.push(
                  context, MaterialPageRoute(builder: (_) => const CameraScreen())),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              ),
              child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Icon(Icons.play_circle_filled_rounded, size: 24),
                SizedBox(width: 10),
                Text('Start Recognition', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
              ]),
            ),
          ),
        ),
        const SizedBox(height: 12),
        Text('Make sure the PC server is running before starting',
            style: TextStyle(color: _inkSoft.withOpacity(0.7), fontSize: 11), textAlign: TextAlign.center),
      ]),
    );
  }
}

class _StepData {
  final String number, title, body;
  final IconData icon;
  final Color color;
  const _StepData(this.number, this.icon, this.title, this.body, this.color);
}