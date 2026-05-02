import 'package:flutter/material.dart';
import '../constants.dart';
import 'camera_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Header ──────────────────────────────
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: kSurface,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: kPrimary.withOpacity(0.3)),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: kPrimary.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: const Icon(Icons.sign_language,
                          color: kPrimary, size: 32),
                    ),
                    const SizedBox(width: 16),
                    const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'SLSL Recognizer',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5,
                          ),
                        ),
                        Text(
                          'Sri Lanka Sign Language',
                          style: TextStyle(
                            color: kAccent,
                            fontSize: 13,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 36),

              // ── Info cards ──────────────────────────
              const Text(
                'How it works',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 16),

              _InfoCard(
                icon: Icons.camera_alt_outlined,
                color: kPrimary,
                title: 'Point camera at hands',
                subtitle: 'Hold your sign for about 2 seconds',
              ),
              const SizedBox(height: 12),
              _InfoCard(
                icon: Icons.auto_awesome,
                color: kSuccess,
                title: 'AI detects the sign',
                subtitle: 'CNN+LSTM model with noise filtering',
              ),
              const SizedBox(height: 12),
              _InfoCard(
                icon: Icons.translate,
                color: kWarning,
                title: 'See the result',
                subtitle: 'Sign name shown with confidence score',
              ),

              const SizedBox(height: 36),

              // ── Signs count ──────────────────────────
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: kSurface.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: kPrimary.withOpacity(0.2)),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _StatItem(
                      value: '${kSignLabels.length}',
                      label: 'Signs',
                      color: kPrimary,
                    ),
                    _divider(),
                    const _StatItem(
                      value: '30',
                      label: 'Frames',
                      color: kAccent,
                    ),
                    _divider(),
                    const _StatItem(
                      value: 'CNN\nLSTM',
                      label: 'Model',
                      color: kSuccess,
                    ),
                  ],
                ),
              ),

              const Spacer(),

              // ── Start button ─────────────────────────
              SizedBox(
                width: double.infinity,
                height: 58,
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => const CameraScreen(),
                      ),
                    );
                  },
                  icon: const Icon(Icons.play_circle_filled, size: 26),
                  label: const Text(
                    'Start Recognition',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: kPrimary,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: 8,
                    shadowColor: kPrimary.withOpacity(0.5),
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  Widget _divider() => Container(
        height: 40,
        width: 1,
        color: Colors.white12,
      );
}

class _InfoCard extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String title;
  final String subtitle;
  const _InfoCard({
    required this.icon,
    required this.color,
    required this.title,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: kSurface.withOpacity(0.6),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.25)),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 26),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title,
                  style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 14)),
              Text(subtitle,
                  style:
                      TextStyle(color: Colors.white.withOpacity(0.6), fontSize: 12)),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String value;
  final String label;
  final Color color;
  const _StatItem(
      {required this.value, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(value,
            textAlign: TextAlign.center,
            style: TextStyle(
                color: color,
                fontSize: 20,
                fontWeight: FontWeight.bold,
                height: 1.2)),
        const SizedBox(height: 4),
        Text(label,
            style: TextStyle(
                color: Colors.white.withOpacity(0.6), fontSize: 12)),
      ],
    );
  }
}