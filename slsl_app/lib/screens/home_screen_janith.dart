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
  late Animation<double>   _fadeAnim;
  late Animation<Offset>   _slideAnim;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 750));
    _fadeAnim  = CurvedAnimation(parent: _controller, curve: Curves.easeOut);
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
      backgroundColor: const Color(0xFF020818),
      body: Stack(
        children: [
          Positioned(top: -40, right: -60, child: _glow(220, const Color(0xFF00B4D8), 0.07)),
          Positioned(bottom: 120, left: -40, child: _glow(180, const Color(0xFF0077B6), 0.06)),
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

  Widget _glow(double size, Color color, double opacity) => Container(
        width: size, height: size,
        decoration: BoxDecoration(shape: BoxShape.circle, boxShadow: [
          BoxShadow(color: color.withOpacity(opacity), blurRadius: size, spreadRadius: size * 0.4),
        ]),
      );

  Widget _buildTopBar(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 12, 20, 0),
      child: Row(children: [
        IconButton(
          icon     : const Icon(Icons.arrow_back_ios_rounded, color: Colors.white54, size: 20),
          onPressed: () => Navigator.maybePop(context),
        ),
        const Spacer(),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            border      : Border.all(color: const Color(0xFF00B4D8).withOpacity(0.35)),
            borderRadius: BorderRadius.circular(20),
            color       : const Color(0xFF00B4D8).withOpacity(0.07),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Container(width: 6, height: 6,
                decoration: const BoxDecoration(shape: BoxShape.circle, color: Color(0xFF00B4D8))),
            const SizedBox(width: 7),
            const Text('AI Module', style: TextStyle(
                color: Color(0xFF90E0EF), fontSize: 11, fontWeight: FontWeight.w600)),
          ]),
        ),
      ]),
    );
  }

  Widget _buildHeroCard() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
      child: Container(
        width  : double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [Color(0xFF023E8A), Color(0xFF03045E)],
          ),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: const Color(0xFF00B4D8).withOpacity(0.25)),
          boxShadow: [BoxShadow(
              color: const Color(0xFF00B4D8).withOpacity(0.1), blurRadius: 30, spreadRadius: 2)],
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color       : const Color(0xFF00B4D8).withOpacity(0.15),
                borderRadius: BorderRadius.circular(16),
                border      : Border.all(color: const Color(0xFF00B4D8).withOpacity(0.3)),
              ),
              child: const Icon(Icons.sign_language, color: Color(0xFF00B4D8), size: 30),
            ),
            const SizedBox(width: 16),
            const Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              SizedBox(height: 2),
              Text('SLSL Recognizer',
                  style: TextStyle(color: Colors.white, fontSize: 22,
                      fontWeight: FontWeight.w800, letterSpacing: -0.3, height: 1.1)),
              SizedBox(height: 5),
              Text('Sri Lanka Sign Language',
                  style: TextStyle(color: Color(0xFF90E0EF), fontSize: 13, fontWeight: FontWeight.w500)),
            ])),
          ]),
          const SizedBox(height: 20),
          Container(height: 1, color: Colors.white.withOpacity(0.07)),
          const SizedBox(height: 18),
          Text(
            'Point your camera at a signer\'s hands and let the AI '
            'identify Sri Lanka Sign Language gestures in real time '
            'with bilingual Sinhala and English output.',
            style: TextStyle(color: Colors.white.withOpacity(0.55), fontSize: 13, height: 1.6),
          ),
        ]),
      ),
    );
  }

  Widget _buildStatsRow() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
      child: Row(children: [
        _statCard('${kSignLabels.length}', 'Signs', Icons.gesture_rounded, const Color(0xFF00B4D8)),
        const SizedBox(width: 10),
        _statCard('30', 'Frames', Icons.video_camera_back_outlined, const Color(0xFF90E0EF)),
        const SizedBox(width: 10),
        _statCard('CNN\nLSTM', 'Model', Icons.memory_rounded, const Color(0xFF06D6A0)),
      ]),
    );
  }

  Widget _statCard(String value, String label, IconData icon, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
        decoration: BoxDecoration(
          color: color.withOpacity(0.06), borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withOpacity(0.18)),
        ),
        child: Column(children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(height: 8),
          Text(value, textAlign: TextAlign.center,
              style: TextStyle(color: color, fontSize: 16, fontWeight: FontWeight.w800,
                  height: 1.15, letterSpacing: -0.3)),
          const SizedBox(height: 4),
          Text(label, style: TextStyle(
              color: Colors.white.withOpacity(0.4), fontSize: 10, fontWeight: FontWeight.w500)),
        ]),
      ),
    );
  }

  Widget _buildHowItWorksLabel() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 16),
      child: Row(children: [
        Container(width: 3, height: 16,
            decoration: BoxDecoration(color: const Color(0xFF00B4D8), borderRadius: BorderRadius.circular(2))),
        const SizedBox(width: 10),
        Text('HOW IT WORKS', style: TextStyle(
            color: Colors.white.withOpacity(0.35), fontSize: 11,
            fontWeight: FontWeight.w700, letterSpacing: 2)),
      ]),
    );
  }

  Widget _buildStepsList() {
    final steps = [
      _StepData('01', Icons.camera_alt_outlined,
          'Point camera at hands', 'Hold your sign clearly in frame with good lighting.', const Color(0xFF00B4D8)),
      _StepData('02', Icons.auto_awesome_rounded,
          'AI detects the sign', 'CNN+LSTM model with velocity-based noise filtering.', const Color(0xFF06D6A0)),
      _StepData('03', Icons.translate_rounded,
          'See the result', 'Sign name in English & Sinhala with confidence score.', const Color(0xFFFFB703)),
    ];
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(children: steps.asMap().entries.map((e) =>
          _buildStepTile(e.value, isLast: e.key == steps.length - 1)).toList()),
    );
  }

  Widget _buildStepTile(_StepData step, {required bool isLast}) {
    return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Column(children: [
        Container(
          width: 36, height: 36,
          decoration: BoxDecoration(
            color : step.color.withOpacity(0.12), shape: BoxShape.circle,
            border: Border.all(color: step.color.withOpacity(0.35), width: 1.5),
          ),
          child: Center(child: Text(step.number, style: TextStyle(
              color: step.color, fontSize: 10, fontWeight: FontWeight.w800))),
        ),
        if (!isLast)
          Container(width: 1, height: 40, margin: const EdgeInsets.symmetric(vertical: 4),
              color: Colors.white.withOpacity(0.07)),
      ]),
      const SizedBox(width: 14),
      Expanded(child: Padding(
        padding: EdgeInsets.only(bottom: isLast ? 0 : 16, top: 6),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color : step.color.withOpacity(0.05), borderRadius: BorderRadius.circular(16),
            border: Border.all(color: step.color.withOpacity(0.15)),
          ),
          child: Row(children: [
            Icon(step.icon, color: step.color, size: 22),
            const SizedBox(width: 12),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(step.title, style: const TextStyle(
                  color: Colors.white, fontSize: 13, fontWeight: FontWeight.w700)),
              const SizedBox(height: 4),
              Text(step.body, style: TextStyle(
                  color: Colors.white.withOpacity(0.5), fontSize: 12, height: 1.4)),
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
          width: double.infinity, height: 58,
          child: ElevatedButton(
            onPressed: () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => const CameraScreen())),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF00B4D8),
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
        const SizedBox(height: 12),
        Text('Make sure the PC server is running before starting',
            style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 11),
            textAlign: TextAlign.center),
      ]),
    );
  }
}

class _StepData {
  final String number, title, body; final IconData icon; final Color color;
  const _StepData(this.number, this.icon, this.title, this.body, this.color);
}