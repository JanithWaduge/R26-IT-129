// lib/screens/sign_playback_screen_hansika.dart
import 'dart:async';
import 'package:flutter/material.dart';
import '../constants.dart';
import '../services/teacher_api_service.dart';

// MediaPipe 21-point hand connections
const List<List<int>> kHandConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [0, 9], [9, 10], [10, 11], [11, 12],  // middle
  [0, 13], [13, 14], [14, 15], [15, 16],// ring
  [0, 17], [17, 18], [18, 19], [19, 20],// pinky
  [5, 9], [9, 13], [13, 17],            // palm
];

class SignPlaybackScreenHansika extends StatefulWidget {
  final String submissionId;
  final String englishWord;
  const SignPlaybackScreenHansika({
    super.key,
    required this.submissionId,
    required this.englishWord,
  });

  @override
  State<SignPlaybackScreenHansika> createState() => _SignPlaybackScreenHansikaState();
}

class _SignPlaybackScreenHansikaState extends State<SignPlaybackScreenHansika> {
  bool _loading = true;
  List<List<double>>? _frames;
  int _currentFrame = 0;
  Timer? _playTimer;
  bool _playing = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _playTimer?.cancel();
    super.dispose();
  }

  Future<void> _load() async {
    final data = await TeacherApiService.getSubmissionDetail(widget.submissionId);
    if (!mounted) return;
    if (data != null && data['keypoint_sequence'] != null) {
      final raw = List<dynamic>.from(data['keypoint_sequence']);
      final frames = raw.map((f) => List<double>.from(f.map((v) => (v as num).toDouble()))).toList();
      setState(() {
        _frames = frames;
        _loading = false;
      });
      _startPlayback();
    } else {
      setState(() => _loading = false);
    }
  }

  void _startPlayback() {
    _playTimer?.cancel();
    _playing = true;
    _playTimer = Timer.periodic(const Duration(milliseconds: 130), (_) {
      if (!mounted || _frames == null) return;
      setState(() {
        _currentFrame = (_currentFrame + 1) % _frames!.length;
      });
    });
  }

  void _togglePlay() {
    if (_playing) {
      _playTimer?.cancel();
      setState(() => _playing = false);
    } else {
      _startPlayback();
      setState(() => _playing = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBackground,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(widget.englishWord, style: const TextStyle(color: kInk, fontWeight: FontWeight.w700)),
        iconTheme: const IconThemeData(color: kInk),
      ),
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator(color: kPrimary))
            : (_frames == null || _frames!.isEmpty)
                ? Center(
                    child: Text('No recorded motion data found for this sign.',
                        style: const TextStyle(color: kInkSoft)),
                  )
                : Column(children: [
                    Expanded(
                      child: Container(
                        margin: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          color: kSurface,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: const Color(0xFFEDEFF5)),
                          boxShadow: [
                            BoxShadow(color: kPrimary.withOpacity(0.08), blurRadius: 22, offset: const Offset(0, 10)),
                          ],
                        ),
                        child: CustomPaint(
                          painter: _HandSkeletonPainter(_frames![_currentFrame]),
                          child: Container(),
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
                      child: Column(children: [
                        SliderTheme(
                          data: SliderTheme.of(context).copyWith(
                            activeTrackColor: kPrimary,
                            inactiveTrackColor: const Color(0xFFEDEFF5),
                            thumbColor: kPrimary,
                            overlayColor: kPrimary.withOpacity(0.15),
                          ),
                          child: Slider(
                            value: _currentFrame.toDouble(),
                            min: 0,
                            max: (_frames!.length - 1).toDouble(),
                            onChanged: (v) {
                              _playTimer?.cancel();
                              _playing = false;
                              setState(() => _currentFrame = v.round());
                            },
                          ),
                        ),
                        Text('Frame ${_currentFrame + 1} / ${_frames!.length}',
                            style: const TextStyle(color: kInkSoft, fontSize: 12)),
                      ]),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(bottom: 24),
                      child: Container(
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: const LinearGradient(colors: [kPrimary, kSecondary]),
                          boxShadow: [
                            BoxShadow(color: kPrimary.withOpacity(0.35), blurRadius: 18, offset: const Offset(0, 8)),
                          ],
                        ),
                        child: FloatingActionButton(
                          backgroundColor: Colors.transparent,
                          elevation: 0,
                          onPressed: _togglePlay,
                          child: Icon(_playing ? Icons.pause : Icons.play_arrow, color: Colors.white),
                        ),
                      ),
                    ),
                  ]),
      ),
    );
  }
}

class _HandSkeletonPainter extends CustomPainter {
  final List<double> keypoints; // 63 values: 21 points x (x,y,z)

  _HandSkeletonPainter(this.keypoints);

  @override
  void paint(Canvas canvas, Size size) {
    if (keypoints.length < 63) return;

    final points = <Offset>[];
    for (int i = 0; i < 21; i++) {
      final x = keypoints[i * 3] * size.width;
      final y = keypoints[i * 3 + 1] * size.height;
      points.add(Offset(x, y));
    }

    final linePaint = Paint()
      ..color = kPrimary
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    final dotPaint = Paint()..color = kSuccess;

    for (final conn in kHandConnections) {
      canvas.drawLine(points[conn[0]], points[conn[1]], linePaint);
    }
    for (final p in points) {
      canvas.drawCircle(p, 5, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _HandSkeletonPainter oldDelegate) => true;
}